
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

