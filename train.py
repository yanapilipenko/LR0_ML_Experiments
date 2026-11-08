
import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
import hashlib
import mlflow

from datetime import datetime
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import yaml
import torch
import torch.nn as nn
import sklearn

from src.dataset import get_dataloaders, SPLIT_PATH
from src.model import DigitsClassifier
from src.utils import set_seed, get_device


# Корневая папка проекта
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"


def get_commit():
    """Получение текущего коммита Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def get_package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def validate_config(config):
    """Проверка корректности конфигурации."""

    model_cfg = config["model"]
    train_cfg = config["train"]

    if model_cfg["n_layers"] not in (1, 2, 3):
        raise ValueError("n_layers должен быть 1, 2 или 3")

    if model_cfg["hidden"] < 32 or model_cfg["hidden"] > 512:
        raise ValueError("hidden должен быть от 32 до 512")

    if not 0 <= model_cfg["dropout"] <= 0.5:
        raise ValueError("dropout должен быть от 0 до 0.5")

    if train_cfg["lr"] <= 0:
        raise ValueError("lr должен быть больше 0")

    if train_cfg["weight_decay"] < 0:
        raise ValueError("weight_decay не может быть отрицательным")

    if train_cfg["batch_size"] not in (32, 64, 128):
        raise ValueError("batch_size должен быть 32, 64 или 128")

    if train_cfg["epochs"] != 30:
        raise ValueError("По заданию количество эпох должно быть 30")

    if config["runtime"]["num_threads"] < 1:
        raise ValueError("num_threads должен быть не меньше 1")


def evaluate(model, loader, criterion, device):
    """Оценка модели на выборке."""

    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * len(y)

            predictions = outputs.argmax(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    return average_loss, y_true, y_pred


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Обучение модели в течение одной эпохи."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += len(y)

    train_loss = total_loss / total
    train_acc = correct / total

    return train_loss, train_acc


def train(config_path, seed_override=None, eval_test=False):

    # Загружаем конфигурацию
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Переопределяем seed из командной строки
    if seed_override is not None:
        config["seed"] = seed_override

    # Проверяем параметры до начала обучения
    validate_config(config)

    seed = config["seed"]
    num_threads = config["runtime"]["num_threads"]

    # Настраиваем воспроизводимость
    set_seed(seed, num_threads)

    device = get_device(config["runtime"]["device"])

    # Создаём отдельную папку запуска
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    start_datetime = datetime.now().astimezone()

    # Контрольная сумма разбиения
    split_hash = hashlib.sha256(
        SPLIT_PATH.read_bytes()
    ).hexdigest()

    metadata = {
        "run_id": run_id,
        "series": config["experiment"],
        "status": "running",
        "start_time": start_datetime.isoformat(),
        "end_time": None,
        "commit": get_commit(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "optuna_version": get_package_version("optuna"),
        "mlflow_version": get_package_version("mlflow"),
        "num_threads": num_threads,
        "seed": seed,
        "split_sha256": split_hash
    }

    save_json(run_dir / "metadata.json", metadata)

    mlflow_started = False

    try:

        # -------------------------------------------------
        # 1. НАСТРОЙКА MLFLOW
        # -------------------------------------------------

        # Локальное хранилище MLflow
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())

        # Эксперимент лабораторной работы
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("LR0_Digits")

        # Создаём отдельный запуск MLflow
        mlflow.start_run(run_name=run_id)
        mlflow_started = True

        # Теги запуска
        mlflow.set_tags({
            "commit": str(metadata["commit"] or "unknown"),
            "series": str(config["experiment"]),
            "seed": str(seed),
            "split_sha256": split_hash,
            "local_run_id": run_id,
            "status": "running"
        })

        # Параметры модели и обучения
        mlflow.log_params({
            "experiment": str(config["experiment"]),
            "split_seed": str(config["split_seed"]),
            "seed": str(seed),

            "model.n_layers": config["model"]["n_layers"],
            "model.hidden": config["model"]["hidden"],
            "model.dropout": config["model"]["dropout"],

            "train.lr": config["train"]["lr"],
            "train.weight_decay": config["train"]["weight_decay"],
            "train.batch_size": config["train"]["batch_size"],
            "train.epochs": config["train"]["epochs"],

            "runtime.device": str(config["runtime"]["device"]),
            "runtime.num_threads": num_threads
        })

        # -------------------------------------------------
        # 2. СОХРАНЕНИЕ КОНФИГУРАЦИИ
        # -------------------------------------------------

        with open(
            run_dir / "config.yaml",
            "w",
            encoding="utf-8"
        ) as file:

            yaml.safe_dump(
                config,
                file,
                allow_unicode=True,
                sort_keys=False
            )

        # -------------------------------------------------
        # 3. ЗАГРУЗКА ДАННЫХ
        # -------------------------------------------------

        train_loader, val_loader, test_loader = get_dataloaders(
            seed=seed,
            batch_size=config["train"]["batch_size"]
        )

        # -------------------------------------------------
        # 4. СОЗДАНИЕ МОДЕЛИ
        # -------------------------------------------------

        model = DigitsClassifier(
            n_layers=config["model"]["n_layers"],
            hidden=config["model"]["hidden"],
            dropout=config["model"]["dropout"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )

        # -------------------------------------------------
        # 5. ОБУЧЕНИЕ
        # -------------------------------------------------

        history_path = run_dir / "history.csv"

        history_fields = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "epoch_time"
        ]

        with open(
            history_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=history_fields
            )

            writer.writeheader()

        print(f"Запуск: {run_id}")
        print(f"Устройство: {device}")
        print(f"Seed: {seed}")

        for epoch in range(1, config["train"]["epochs"] + 1):

            epoch_start = time.perf_counter()

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            val_loss, val_true, val_pred = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            val_acc = sum(
                a == b for a, b in zip(val_true, val_pred)
            ) / len(val_true)

            epoch_time = time.perf_counter() - epoch_start

            # Сохраняем историю локально
            with open(
                history_path,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=history_fields
                )

                writer.writerow({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": epoch_time
                })

            # Записываем метрики эпохи в MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time": epoch_time
            }, step=epoch)

            print(
                f"Epoch {epoch:02d}/30 | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        # -------------------------------------------------
        # 6. ПРЕДСКАЗАНИЯ НА VALIDATION
        # -------------------------------------------------

        val_loss, val_true, val_pred = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        with open(SPLIT_PATH, "r", encoding="utf-8") as file:
            split_data = json.load(file)

        predictions = []

        for index, true, pred in zip(
            split_data["val"], val_true, val_pred
        ):
            predictions.append({
                "split": "val",
                "index": index,
                "y_true": true,
                "y_pred": pred
            })

        # Test используется только при наличии флага
        if eval_test:

            test_loss, test_true, test_pred = evaluate(
                model,
                test_loader,
                criterion,
                device
            )

            for index, true, pred in zip(
                split_data["test"], test_true, test_pred
            ):
                predictions.append({
                    "split": "test",
                    "index": index,
                    "y_true": true,
                    "y_pred": pred
                })

        # -------------------------------------------------
        # 7. СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ
        # -------------------------------------------------

        predictions_path = run_dir / "predictions.csv"

        with open(
            predictions_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            fields = ["split", "index", "y_true", "y_pred"]

            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            writer.writeheader()
            writer.writerows(predictions)

        # -------------------------------------------------
        # 8. РАСЧЁТ ИТОГОВЫХ МЕТРИК
        # -------------------------------------------------

        with open(history_path, "r", encoding="utf-8") as file:
            history_rows = list(csv.DictReader(file))

        with open(predictions_path, "r", encoding="utf-8") as file:
            prediction_rows = list(csv.DictReader(file))

        val_rows = [
            row for row in prediction_rows
            if row["split"] == "val"
        ]

        if not history_rows or not val_rows:
            raise ValueError(
                "История обучения или validation-предсказания пусты"
            )

        # Accuracy рассчитываем по сохранённым предсказаниям
        final_val_acc = sum(
            int(row["y_true"]) == int(row["y_pred"])
            for row in val_rows
        ) / len(val_rows)

        final_val_loss = float(history_rows[-1]["val_loss"])

        training_time = sum(
            float(row["epoch_time"])
            for row in history_rows
        )

        test_rows = [
            row for row in prediction_rows
            if row["split"] == "test"
        ]

        final_test_acc = None

        if test_rows:
            final_test_acc = sum(
                int(row["y_true"]) == int(row["y_pred"])
                for row in test_rows
            ) / len(test_rows)

        metrics = {
            "val_acc": final_val_acc,
            "val_loss": final_val_loss,
            "training_time": training_time,
            "test_acc": final_test_acc
        }

        save_json(run_dir / "metrics.json", metrics)

        # -------------------------------------------------
        # 9. ЗАВЕРШЕНИЕ MLFLOW
        # -------------------------------------------------

        metadata["status"] = "completed"
        metadata["end_time"] = datetime.now().astimezone().isoformat()

        save_json(run_dir / "metadata.json", metadata)

        # Итоговые метрики
        mlflow.log_metrics({
            "final_val_acc": final_val_acc,
            "final_val_loss": final_val_loss,
            "training_time": training_time
        })

        if final_test_acc is not None:
            mlflow.log_metric("test_acc", final_test_acc)

        # Обновляем статус
        mlflow.set_tag("status", "completed")

        # Загружаем все файлы запуска как артефакты
        mlflow.log_artifacts(str(run_dir))

        # Завершаем запуск
        mlflow.end_run(status="FINISHED")
        mlflow_started = False

        # -------------------------------------------------
        # 10. ВЫВОД РЕЗУЛЬТАТОВ
        # -------------------------------------------------

        print("\nОбучение завершено!")
        print(f"Validation accuracy: {final_val_acc:.4f}")

        if final_test_acc is not None:
            print(f"Test accuracy: {final_test_acc:.4f}")

        print(f"Training time: {training_time:.2f} секунд")
        print(f"Результаты сохранены: {run_dir}")

    except Exception:

        metadata["status"] = "failed"
        metadata["end_time"] = datetime.now().astimezone().isoformat()

        save_json(run_dir / "metadata.json", metadata)

        with open(
            run_dir / "traceback.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(traceback.format_exc())

        # Сохраняем неудачный запуск в MLflow
        if mlflow_started and mlflow.active_run() is not None:

            try:
                mlflow.set_tag("status", "failed")
                mlflow.log_artifacts(str(run_dir))

            except Exception as mlflow_error:
                print(
                    f"Ошибка журналирования MLflow: {mlflow_error}"
                )

            finally:
                mlflow.end_run(status="FAILED")
                mlflow_started = False

        raise


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--eval-test",
        action="store_true"
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    train(
        config_path=config_path,
        seed_override=args.seed,
        eval_test=args.eval_test
    )


if __name__ == "__main__":
    main()
