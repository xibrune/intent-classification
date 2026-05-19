"""
Запуск из корня репозитория:
    python src/predict.py --text "What's my balance?" --method finetuning
    python src/predict.py --text "What's my balance?" --method zeroshot
    python src/predict.py --text "What's my balance?" --method fewshot --n_shots 5 --strategy central
"""

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import torch

# src/ в путь
sys.path.insert(0, os.path.dirname(__file__))
from intent_mapping import ID_TO_INTENT, INTENT_TO_ID

# Пути (относительно корня репозитория)
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(ROOT, 'results', 'models', 'finetuned_distilbert')
TRAIN_CSV     = os.path.join(ROOT, 'data', 'splits', 'train.csv')
FLAN_MODEL_ID = 'google/flan-t5-base'

INTENT_NAMES  = [ID_TO_INTENT[i] for i in range(len(ID_TO_INTENT))]  # список 151 интента
SEED          = 42

# ── Устройство ───────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ПРЕДИКТОР 1: Fine-Tuning (DistilBERT)
def predict_finetuning(text: str) -> str:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = get_device()
    print(f'[Fine-Tuning] Устройство: {device}')
    print(f'[Fine-Tuning] Загрузка модели из {MODEL_DIR} ...')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_id = int(logits.argmax(dim=-1).item())
    return ID_TO_INTENT[pred_id]


# ПРЕДИКТОР 2: Zero-Shot (Flan-T5)
def predict_zeroshot(text, batch_size = 32):
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    device = get_device()
    print(f'[Zero-Shot] Устройство: {device}')
    print(f'[Zero-Shot] Загрузка {FLAN_MODEL_ID} ...')

    tokenizer = T5Tokenizer.from_pretrained(FLAN_MODEL_ID)
    model     = T5ForConditionalGeneration.from_pretrained(FLAN_MODEL_ID)
    model.to(device).eval()

    yes_id = tokenizer('Yes', add_special_tokens=False).input_ids[0]
    no_id  = tokenizer('No',  add_special_tokens=False).input_ids[0]

    prompts = [
        f'Does the user want to {intent}? text: {text}'
        for intent in INTENT_NAMES
    ]

    scores = []
    print(f'[Zero-Shot] Оценка {len(INTENT_NAMES)} интентов (batch_size={batch_size}) ...')

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True,
        ).to(device)

        # Один шаг декодера — получаем логиты первого токена
        decoder_input_ids = torch.full(
            (len(batch), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)

        logits   = outputs.logits[:, 0, :]                    # (batch, vocab)
        probs    = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)[:, 0]
        scores.extend(probs.cpu().tolist())

    return INTENT_NAMES[int(np.argmax(scores))]


# ПРЕДИКТОР 3: Few-Shot (Flan-T5)
def _sample_random(intent_df: pd.DataFrame, n: int) -> list[str]:
    return intent_df['text'].sample(n=n, random_state=SEED).tolist()


def _sample_central_preloaded(texts: list, n: int, st_model) -> list:
    embs     = st_model.encode(texts, normalize_embeddings=True)
    centroid = embs.mean(axis=0)
    dists    = np.linalg.norm(embs - centroid, axis=1)
    idxs     = np.argsort(dists)[:n]
    return [texts[i] for i in idxs]


def _sample_diverse_preloaded(texts: list, n: int, st_model) -> list:
    embs     = st_model.encode(texts, normalize_embeddings=True)
    selected = [0]
    for _ in range(n - 1):
        remaining = [i for i in range(len(texts)) if i not in selected]
        min_sims  = np.array([
            max(float(np.dot(embs[i], embs[s])) for s in selected)
            for i in remaining
        ])
        selected.append(remaining[int(np.argmin(min_sims))])
    return [texts[i] for i in selected]


def predict_fewshot(text: str, n_shots: int = 10, strategy: str = 'diverse') -> str:
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    device = get_device()
    print(f'[Few-Shot] Устройство: {device}')
    print(f'[Few-Shot] Стратегия: {strategy}, n_shots: {n_shots}')
    print(f'[Few-Shot] Загрузка {FLAN_MODEL_ID} ...')

    tokenizer = T5Tokenizer.from_pretrained(FLAN_MODEL_ID)
    model     = T5ForConditionalGeneration.from_pretrained(FLAN_MODEL_ID)
    model.to(device).eval()

    train = pd.read_csv(TRAIN_CSV)

    # Загружаем sentence-transformers один раз
    st_model = None
    if strategy in ('central', 'diverse'):
        from sentence_transformers import SentenceTransformer
        print('[Few-Shot] Загрузка sentence-transformers ...')
        st_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Строим все 151 промпт
    print('[Few-Shot] Подготовка промптов ...')
    prompts = []
    for intent in INTENT_NAMES:
        intent_df = train[train['intent'] == INTENT_TO_ID[intent]]
        n         = min(n_shots, len(intent_df))
        texts     = intent_df['text'].tolist()

        if n == 0:
            examples = []
        elif strategy == 'random':
            examples = _sample_random(intent_df, n)
        elif strategy == 'central':
            examples = _sample_central_preloaded(texts, n, st_model)
        else:
            examples = _sample_diverse_preloaded(texts, n, st_model)

        examples_str = '\n'.join(f'- {ex}' for ex in examples)
        prompt = (
            f'Examples of intent "{intent}":\n'
            f'{examples_str}\n'
            f'Now classify: "{text}"\n'
            f'Is this intent "{intent}"? Yes or No.'
        )
        prompts.append(prompt)

    # ── Батчевый inference ───────────────────────────────────────────────────
    yes_id = tokenizer('Yes', add_special_tokens=False).input_ids[0]
    no_id  = tokenizer('No',  add_special_tokens=False).input_ids[0]

    batch_size = 16
    scores     = []
    print(f'[Few-Shot] Оценка {len(INTENT_NAMES)} интентов (batch_size={batch_size}) ...')

    for i in range(0, len(prompts), batch_size):
        batch  = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True,
        ).to(device)

        decoder_input_ids = torch.full(
            (len(batch), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)

        logits = outputs.logits[:, 0, :]
        probs  = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)[:, 0]
        scores.extend(probs.cpu().tolist())

    return INTENT_NAMES[int(np.argmax(scores))]


def main():
    parser = argparse.ArgumentParser(
        description='Классификация интента тремя методами (CLINC150)'
    )
    parser.add_argument('--text',     required=True,  type=str)
    parser.add_argument('--method',   required=True,
                        choices=['finetuning', 'zeroshot', 'fewshot'])
    parser.add_argument('--n_shots',  default=10,     type=int)
    parser.add_argument('--strategy', default='diverse',
                        choices=['random', 'central', 'diverse'])
    args = parser.parse_args()

    print(f'\nТекст запроса : «{args.text}»')
    print(f'Метод         : {args.method}')

    start = time.time()

    if args.method == 'finetuning':
        result = predict_finetuning(args.text)
    elif args.method == 'zeroshot':
        result = predict_zeroshot(args.text)
    elif args.method == 'fewshot':
        print(f'n_shots       : {args.n_shots}')
        print(f'strategy      : {args.strategy}')
        result = predict_fewshot(args.text, args.n_shots, args.strategy)

    elapsed = time.time() - start
    print(f'\n>>> Предсказанный интент: {result}')
    print(f'    Время inference     : {elapsed:.1f} сек.')


if __name__ == '__main__':
    main()