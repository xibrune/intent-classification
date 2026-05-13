import random
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

dataset = load_dataset("clinc_oos", "plus")

dfs = []
for split_name in ["train", "validation", "test"]:
    df = dataset[split_name].to_pandas()
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

# Разбиение
train_val, test = train_test_split(
    full_df,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
    stratify=full_df["intent"]
)

val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

train, val = train_test_split(
    train_val,
    test_size=val_ratio_adjusted,
    random_state=RANDOM_SEED,
    stratify=train_val["intent"]
)

# Проверка
assert set(train["intent"].unique()) == set(val["intent"].unique()) == set(test["intent"].unique())
assert len(set(train["text"]) & set(test["text"])) == 0

# Сохранение
train.to_csv("splits/train.csv", index=False)
val.to_csv("splits/val.csv",     index=False)
test.to_csv("splits/test.csv",   index=False)

print("Сохранено: train.csv, val.csv, test.csv")