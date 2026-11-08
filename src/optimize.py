
import argparse
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import DigitsClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed, num_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(num_threads)


def objective(trial, config):
    # Для всех испытаний используем одинаковую начальную случайность
    seed = config["seed"]
    num_threads = config["runtime"]["num_threads"]

    set_seed(seed, num_threads)

    device = torch.device(config["runtime"]["device"])

    # Подбираемые гиперпараметры
    lr = trial.suggest_float(
        "lr", 0.0001, 0.1, log=True
    )

    n_layers = trial.suggest_int(
        "n_layers", 1, 3
    )

    hidden = trial.suggest_int(
        "hidden", 32, 512, log=True
    )

    dropout = trial.suggest_float(
        "dropout", 0.0, 0.5
    )

    weight_decay = trial.suggest_float(
        "weight_decay", 0.000001, 0.01, log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128]
    )

    # Загружаем одни и те же выборки
    train_loader, val_loader, _ = get_dataloaders(
        seed=seed,
        batch_size=batch_size
    )

    # Создаём модель
    model = DigitsClassifier(
        n_layers=n_layers,
        hidden=hidden,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    epochs = config["search"]["epochs"]

    # Обучение
    for epoch in range(epochs):

        model.train()

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

        # Проверка на validation
        model.eval()

        val_loss = 0.0
        total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item() * X.size(0)
                total += X.size(0)

        val_loss /= total

        # Сохраняем промежуточный результат
        trial.report(val_loss, epoch)

        trial.set_user_attr("executed_epochs", epoch + 1)

        # Проверяем, нужно ли остановить испытание
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/search.yaml"
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    search = config["search"]

    # Алгоритм поиска
    sampler = optuna.samplers.TPESampler(
        seed=search["sampler_seed"]
    )

    # Досрочная остановка неудачных испытаний
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=search["pruner"]["n_startup_trials"],
        n_warmup_steps=search["pruner"]["n_warmup_steps"]
    )

    # Создание исследования
    study = optuna.create_study(
        study_name=search["study_name"],
        storage=search["storage"],
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    print("Начинаем подбор гиперпараметров...")

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=search["n_trials"],
        n_jobs=1
    )

    print("\nПодбор завершён!")
    print("Количество испытаний:", len(study.trials))

    print("\nЛучшее значение val_loss:")
    print(study.best_value)

    print("\nЛучшие гиперпараметры:")

    for name, value in study.best_params.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
