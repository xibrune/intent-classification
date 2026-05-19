"""
Few-Shot — логика выбора примеров для промптов.

Модуль содержит три стратегии отбора n примеров одного интента из train:

  random   — случайный выбор (baseline, никакой семантики)
  central  — n примеров, ближайших к центроиду класса в пространстве эмбеддингов
             (самые "типичные" представители интента)
  diverse  — n максимально непохожих друг на друга примеров
             (жадный farthest-point sampling по косинусному расстоянию)

Эмбеддинги считаются моделью sentence-transformers (all-MiniLM-L6-v2).
Используется из run_fewshot.py через диспетчер select_examples().
"""

import numpy as np
RANDOM_SEED = 42

# Стратегия 1: random — случайный выбор
def select_random(texts, n, rng):
    if len(texts) <= n:
        # примеров меньше, чем просят — возвращаем все, что есть
        return list(texts)
    idx = rng.choice(len(texts), size=n, replace=False)
    return [texts[i] for i in idx]

# Стратегия 2: central — ближайшие к центроиду класса
def select_central(texts, n, embedder):
    """
    Выбирает n примеров, наиболее близких к центроиду класса.

    Идея: центроид — это "усреднённый смысл" интента. Примеры рядом с ним
    самые типичные и однозначные, без редких формулировок.
    """
    if len(texts) <= n:
        return list(texts)

    # normalize_embeddings=True -> длина каждого вектора = 1,
    # поэтому скалярное произведение = косинусная близость
    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb)

    centroid = emb.mean(axis=0)              # "средний" вектор класса
    sims = emb @ centroid                    # косинусная близость каждого примера к центроиду
    top_idx = np.argsort(-sims)[:n]          # n самых близких (сортировка по убыванию)
    return [texts[i] for i in top_idx]


# Стратегия 3: diverse — максимально разнообразные примеры
def select_diverse(texts, n, embedder):
    """
    Выбирает n максимально непохожих друг на друга примеров.

    Алгоритм — жадный farthest-point sampling:
      1. Первым берём пример, ближайший к центроиду (надёжная "якорная" точка).
      2. Каждый следующий — тот, который максимально далёк от уже выбранных
         (максимизируем минимальное косинусное расстояние до выбранного набора).

    Так набор покрывает разные формулировки интента, а не одну его область.
    """
    if len(texts) <= n:
        return list(texts)

    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb)

    # матрица косинусной близости всех пар (векторы нормированы)
    sim_matrix = emb @ emb.T

    # первый пример — ближайший к центроиду
    centroid = emb.mean(axis=0)
    sims_to_centroid = emb @ centroid
    first = int(np.argmax(sims_to_centroid))
    selected = [first]

    # жадно добавляем самый "далёкий" пример, пока не наберём n
    while len(selected) < n:
        # для каждого примера — максимальная близость к уже выбранным
        # (чем она меньше, тем пример дальше от набора)
        max_sim_to_selected = sim_matrix[:, selected].max(axis=1)
        # уже выбранные исключаем из кандидатов
        max_sim_to_selected[selected] = np.inf
        next_idx = int(np.argmin(max_sim_to_selected))
        selected.append(next_idx)

    return [texts[i] for i in selected]


# Диспетчер — единая точка входа для run_fewshot.py
def select_examples(texts, n, strategy, embedder=None, seed=RANDOM_SEED):
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return select_random(texts, n, rng)

    if strategy == "central":
        if embedder is None:
            raise ValueError("Стратегия 'central' требует embedder (sentence-transformers).")
        return select_central(texts, n, embedder)

    if strategy == "diverse":
        if embedder is None:
            raise ValueError("Стратегия 'diverse' требует embedder (sentence-transformers).")
        return select_diverse(texts, n, embedder)

    raise ValueError(
        f"Неизвестная стратегия: {strategy!r}. "
        f"Доступны: 'random', 'central', 'diverse'."
    )

if __name__ == "__main__":
    # Фейковый "embedder" с тем же интерфейсом .encode(), что у sentence-transformers,
    # чтобы проверить математику central/diverse без скачивания модели.
    class _FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            rng = np.random.default_rng(0)
            vecs = rng.normal(size=(len(texts), 8))
            if normalize_embeddings:
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    demo_texts = [f"example sentence number {i}" for i in range(20)]
    fake = _FakeEmbedder()

    for strat in ["random", "central", "diverse"]:
        for k in [1, 3, 5, 10]:
            picked = select_examples(demo_texts, k, strat, embedder=fake, seed=RANDOM_SEED)
            assert len(picked) == k, f"{strat}/{k}: ожидалось {k}, получено {len(picked)}"
            assert len(set(picked)) == k, f"{strat}/{k}: есть дубликаты"
        print(f"  [ok] стратегия '{strat}' прошла проверку для n_shots = 1, 3, 5, 10")

    # граничный случай: примеров меньше, чем n_shots
    short = ["only", "three", "items"]
    assert select_examples(short, 10, "random") == short
    print("  [ok] граничный случай (примеров меньше n_shots) обработан")
    print("\nsample_strategies.py: все проверки пройдены.")