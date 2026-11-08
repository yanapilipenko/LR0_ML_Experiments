
import json
import hashlib
from pathlib import Path

import sklearn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# Параметры разбиения
SPLIT_SEED = 0

# Пути к файлам
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SPLIT_PATH = DATA_DIR / "split.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def create_dataset_split():
    # Создаём папку data, если её ещё нет
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем набор данных Digits
    digits = load_digits()

    X = digits.data
    y = digits.target

    indices = np.arange(len(X))

    # Сначала выделяем тестовую выборку (20%)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=y
    )

    # Из оставшихся данных выделяем validation (25%)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=SPLIT_SEED,
        stratify=y[train_val_idx]
    )

    # Проверяем размеры выборок
    assert len(train_idx) == 1077
    assert len(val_idx) == 360
    assert len(test_idx) == 360

    # Проверяем отсутствие пересечений
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # Проверяем, что использованы все изображения
    all_indices = np.concatenate(
        [train_idx, val_idx, test_idx]
    )

    assert len(all_indices) == 1797
    assert len(np.unique(all_indices)) == 1797

    # Проверяем наличие всех классов в каждой выборке
    for name, idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx)
    ]:
        classes = np.unique(y[idx])
        assert np.array_equal(classes, np.arange(10))

        print(f"{name}: {len(idx)} изображений")
        print(f"Классы: {classes.tolist()}")

    # Формируем структуру split.json
    split_data = {
        "split_seed": SPLIT_SEED,
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }

    # Сохраняем индексы
    with open(SPLIT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            split_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Вычисляем SHA-256 файла split.json
    sha256 = hashlib.sha256(
        SPLIT_PATH.read_bytes()
    ).hexdigest()

    # Формируем манифест данных
    manifest = {
        "source": "scikit-learn load_digits",
        "sklearn_version": sklearn.__version__,
        "split_rule": (
            "Stratified split: test=20% first, "
            "then val=25% of remaining data"
        ),
        "split_seed": SPLIT_SEED,
        "sizes": {
            "total": len(X),
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        },
        "sha256": sha256
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nРазбиение успешно создано!")
    print(f"SHA-256: {sha256}")
    print(f"Файл индексов: {SPLIT_PATH}")
    print(f"Манифест: {MANIFEST_PATH}")


if __name__ == "__main__":
    create_dataset_split()


import torch
from torch.utils.data import TensorDataset, DataLoader


def get_dataloaders(seed=0, batch_size=64):
    """
    Загружает train, val и test
    и создаёт DataLoader для каждой выборки.
    """

    # Загружаем исходные данные
    digits = load_digits()

    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    # Масштабируем признаки от 0 до 1
    # Исходный диапазон: 0...16
    X = X / 16.0

    # Загружаем индексы разбиения
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    train_idx = np.array(split_data["train"])
    val_idx = np.array(split_data["val"])
    test_idx = np.array(split_data["test"])

    # Создаём TensorDataset
    train_dataset = TensorDataset(
        torch.tensor(X[train_idx]),
        torch.tensor(y[train_idx])
    )

    val_dataset = TensorDataset(
        torch.tensor(X[val_idx]),
        torch.tensor(y[val_idx])
    )

    test_dataset = TensorDataset(
        torch.tensor(X[test_idx]),
        torch.tensor(y[test_idx])
    )

    # Фиксируем генератор перемешивания
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Создаём DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
