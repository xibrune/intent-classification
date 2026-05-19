"""
run_fewshot.py

Few-Shot = показать модели несколько примеров интента прямо в промпте,
не обучая её. Проверяем, как качество растёт с числом примеров (n_shots)
и зависит от способа их выбора (strategy).

МАТРИЦА ЭКСПЕРИМЕНТОВ
  n_shots  ∈ {1, 3, 5, 10}
  strategy ∈ {random, central, diverse}
  = 12 комбинаций

КАК КЛАССИФИЦИРУЕМ
  Для каждого тестового запроса перебираем все интенты 
  и по каждому задаём бинарный вопрос с few-shot примерами:
      Examples of intent "weather":
      - what's the forecast for tomorrow
      - is it going to rain today
      - how hot is it outside
      Now classify: "do i need an umbrella"
      Is this intent "weather"? Yes or No.
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from sample_strategies import select_examples


RANDOM_SEED = 42

N_SHOTS_GRID = [1, 3, 5, 10]
STRATEGY_GRID = ["random", "central", "diverse"]

LLM_MODEL = "google/flan-t5-base"           # генеративная модель для классификации
EMBED_MODEL = "all-MiniLM-L6-v2"            # эмбеддинги для central / diverse

DEBUG_N_INTENTS = 10                        # сколько интентов в режиме --debug
DEBUG_N_TEST = 30                           # сколько тестовых примеров в режиме --debug



def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Маппинг id
ID_TO_INTENT = {
    0: "restaurant_reviews",
    1: "nutrition_info",
    2: "account_blocked",
    3: "oil_change_how",
    4: "time",
    5: "weather",
    6: "redeem_rewards",
    7: "interest_rate",
    8: "gas_type",
    9: "accept_reservations",
    10: "smart_home",
    11: "user_name",
    12: "report_lost_card",
    13: "repeat",
    14: "whisper_mode",
    15: "what_are_your_hobbies",
    16: "order",
    17: "jump_start",
    18: "schedule_meeting",
    19: "meeting_schedule",
    20: "freeze_account",
    21: "what_song",
    22: "meaning_of_life",
    23: "restaurant_reservation",
    24: "traffic",
    25: "make_call",
    26: "text",
    27: "bill_balance",
    28: "improve_credit_score",
    29: "change_language",
    30: "no",
    31: "measurement_conversion",
    32: "timer",
    33: "flip_coin",
    34: "do_you_have_pets",
    35: "balance",
    36: "tell_joke",
    37: "last_maintenance",
    38: "exchange_rate",
    39: "uber",
    40: "car_rental",
    41: "credit_limit",
    42: "oos",
    43: "shopping_list",
    44: "expiration_date",
    45: "routing",
    46: "meal_suggestion",
    47: "tire_change",
    48: "todo_list",
    49: "card_declined",
    50: "rewards_balance",
    51: "change_accent",
    52: "vaccines",
    53: "reminder_update",
    54: "food_last",
    55: "change_ai_name",
    56: "bill_due",
    57: "who_do_you_work_for",
    58: "share_location",
    59: "international_visa",
    60: "calendar",
    61: "translate",
    62: "carry_on",
    63: "book_flight",
    64: "insurance_change",
    65: "todo_list_update",
    66: "timezone",
    67: "cancel_reservation",
    68: "transactions",
    69: "credit_score",
    70: "report_fraud",
    71: "spending_history",
    72: "directions",
    73: "spelling",
    74: "insurance",
    75: "what_is_your_name",
    76: "reminder",
    77: "where_are_you_from",
    78: "distance",
    79: "payday",
    80: "flight_status",
    81: "find_phone",
    82: "greeting",
    83: "alarm",
    84: "order_status",
    85: "confirm_reservation",
    86: "cook_time",
    87: "damaged_card",
    88: "reset_settings",
    89: "pin_change",
    90: "replacement_card_duration",
    91: "new_card",
    92: "roll_dice",
    93: "income",
    94: "taxes",
    95: "date",
    96: "who_made_you",
    97: "pto_request",
    98: "tire_pressure",
    99: "how_old_are_you",
    100: "rollover_401k",
    101: "pto_request_status",
    102: "how_busy",
    103: "application_status",
    104: "recipe",
    105: "calendar_update",
    106: "play_music",
    107: "yes",
    108: "direct_deposit",
    109: "credit_limit_change",
    110: "gas",
    111: "pay_bill",
    112: "ingredients_list",
    113: "lost_luggage",
    114: "goodbye",
    115: "what_can_i_ask_you",
    116: "book_hotel",
    117: "are_you_a_bot",
    118: "next_song",
    119: "change_speed",
    120: "plug_type",
    121: "maybe",
    122: "w2",
    123: "oil_change_when",
    124: "thank_you",
    125: "shopping_list_update",
    126: "pto_balance",
    127: "order_checks",
    128: "travel_alert",
    129: "fun_fact",
    130: "sync_device",
    131: "schedule_maintenance",
    132: "apr",
    133: "transfer",
    134: "ingredient_substitution",
    135: "calories",
    136: "current_location",
    137: "international_fees",
    138: "calculator",
    139: "definition",
    140: "next_holiday",
    141: "update_playlist",
    142: "mpg",
    143: "min_payment",
    144: "change_user_name",
    145: "restaurant_suggestion",
    146: "travel_notification",
    147: "cancel",
    148: "pto_used",
    149: "travel_suggestion",
    150: "change_volume",
}


def humanize(intent_name):
    """
    oil_change_how -> 'oil change how'. Так формулировка читается естественнее
    для языковой модели, чем строка с подчёркиваниями.
    """
    return intent_name.replace("_", " ")


# Построение few-shot промпта
def build_prompt(intent_name, examples, query):
    human = humanize(intent_name)
    lines = [f'Examples of intent "{human}":']
    for ex in examples:
        lines.append(f"- {ex}")
    lines.append(f'Now classify: "{query}"')
    lines.append(f'Is this intent "{human}"? Yes or No.')
    return "\n".join(lines)


# Few-shot примеры по каждому интенту для заданной (n_shots, strategy)
def build_example_bank(train_by_intent, n_shots, strategy, embedder):
    bank = {}
    for intent_id, texts in train_by_intent.items():
        # seed зависит от интента -> разные интенты получают разные примеры,
        # но прогон полностью воспроизводим
        seed = RANDOM_SEED + intent_id
        bank[intent_id] = select_examples(
            texts, n_shots, strategy, embedder=embedder, seed=seed
        )
    return bank


# Оценка вероятности ответа "yes" батчами
def score_yes(prompts, model, tokenizer, device, yes_id, no_id, batch_size=16):
    import torch
    probs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        # стартовый токен декодера — модель ещё ничего не сгенерировала
        dec_start = torch.full(
            (len(batch), 1), model.config.decoder_start_token_id, device=device
        )
        with torch.no_grad():
            out = model(**enc, decoder_input_ids=dec_start)
        first_logits = out.logits[:, 0, :]                  # логиты первого токена ответа
        yn = torch.softmax(first_logits[:, [yes_id, no_id]], dim=-1)
        probs.extend(yn[:, 0].tolist())                     # P(yes)
    return probs


# Прогон одной комбинации (n_shots, strategy) по всему тесту
def run_combination(n_shots, strategy, test_df, train_by_intent, intent_ids,
                    id_to_intent, model, tokenizer, device, yes_id, no_id,
                    embedder, batch_size):
    
    # 1) выбираем few-shot примеры для каждого интента
    example_bank = build_example_bank(train_by_intent, n_shots, strategy, embedder)

    y_true, y_pred = [], []
    t0 = time.time()

    # 2) для каждого тестового запроса перебираем все интенты
    for row_i, (_, row) in enumerate(test_df.iterrows()):
        query = row["text"]
        true_id = int(row["intent"])

        prompts = [
            build_prompt(id_to_intent[iid], example_bank[iid], query)
            for iid in intent_ids
        ]
        yes_probs = score_yes(
            prompts, model, tokenizer, device, yes_id, no_id, batch_size
        )
        # интент с максимальной вероятностью "yes"
        best_pos = int(np.argmax(yes_probs))
        pred_id = intent_ids[best_pos]

        y_true.append(true_id)
        y_pred.append(pred_id)

        # прогресс с ETA каждые 50 примеров — прогон длинный, без этого
        # непонятно, работает скрипт или завис
        if (row_i + 1) % 50 == 0 or (row_i + 1) == len(test_df):
            elapsed = time.time() - t0
            speed = (row_i + 1) / elapsed
            eta = (len(test_df) - row_i - 1) / speed
            print(
                f"    [{strategy}/{n_shots}-shot] "
                f"{row_i + 1}/{len(test_df)}  "
                f"({speed:.2f} примеров/с, ETA {eta/60:.1f} мин)"
            )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, y_pred


# main
def run(args):
    set_seed(RANDOM_SEED)
    os.makedirs(args.results_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print(f"  train: {len(train_df)} строк, test: {len(test_df)} строк")

    id_to_intent = ID_TO_INTENT
    print(f"  маппинг интентов: {len(id_to_intent)} классов")

    # все интенты, которые встречаются в тесте
    intent_ids = sorted(test_df["intent"].unique().tolist())

    # train, сгруппированный по интенту: {intent_id: [тексты]}
    train_by_intent = {
        iid: grp["text"].tolist()
        for iid, grp in train_df.groupby("intent")
    }

    # режим отладки
    if args.debug:
        intent_ids = intent_ids[:DEBUG_N_INTENTS]
        test_df = test_df[test_df["intent"].isin(intent_ids)].head(DEBUG_N_TEST)
        print(f"  [DEBUG] урезано до {len(intent_ids)} интентов и {len(test_df)} примеров")

    device = get_device()
    print(f"Device: {device}")

    print(f"Загружаю LLM: {LLM_MODEL} ...")
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL).to(device)
    model.eval()

    # id токенов "yes" / "no" — берём первый токен слова
    yes_id = tokenizer.encode("yes")[0]
    no_id = tokenizer.encode("no")[0]

    # эмбеддер нужен только если есть стратегии central / diverse
    strategies = STRATEGY_GRID
    embedder = None
    if any(s in ("central", "diverse") for s in strategies):
        print(f"Загружаю эмбеддер: {EMBED_MODEL} ...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL, device=device)

    # матрица экспериментов
    total = len(N_SHOTS_GRID) * len(strategies)
    print(f"\nЗапускаю матрицу: {len(N_SHOTS_GRID)} n_shots x "
          f"{len(strategies)} стратегий = {total} комбинаций\n")

    metrics_rows = []
    predictions = {"text": test_df["text"].tolist(),
                   "true_intent": test_df["intent"].tolist()}

    metrics_path = os.path.join(args.results_dir, "exp03_metrics.csv")
    pred_path = os.path.join(args.results_dir, "exp03_predictions.csv")

    combo_i = 0
    for n_shots in N_SHOTS_GRID:
        for strategy in strategies:
            combo_i += 1
            print(f"[{combo_i}/{total}] strategy={strategy}, n_shots={n_shots}")
            acc, macro_f1, y_pred = run_combination(
                n_shots, strategy, test_df, train_by_intent, intent_ids,
                id_to_intent, model, tokenizer, device, yes_id, no_id,
                embedder, args.batch_size,
            )
            print(f"    -> accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
            metrics_rows.append({
                "n_shots": n_shots, "strategy": strategy,
                "f1": macro_f1, "accuracy": acc,
            })
            predictions[f"pred_{strategy}_{n_shots}shot"] = y_pred

            # Промежуточное сохранение после КАЖDOЙ комбинации.
            pd.DataFrame(
                metrics_rows, columns=["n_shots", "strategy", "f1", "accuracy"]
            ).to_csv(metrics_path, index=False)
            pd.DataFrame(predictions).to_csv(pred_path, index=False)
            print(f"    промежуточно сохранено -> {metrics_path} "
                  f"({combo_i}/{total} комбинаций)\n")

    metrics_df = pd.DataFrame(metrics_rows, columns=["n_shots", "strategy", "f1", "accuracy"])
    print("Готово.")
    print(f"  метрики:      {metrics_path}")
    print(f"  предсказания: {pred_path}")
    print("\nИтоговая таблица:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-Shot с Flan-T5"
    )
    parser.add_argument(
        "--train_path", type=str, default="data/splits/train.csv"
    )
    parser.add_argument(
        "--test_path", type=str, default="data/splits/test.csv"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--debug", action="store_true"
    )
    args = parser.parse_args()
    run(args)