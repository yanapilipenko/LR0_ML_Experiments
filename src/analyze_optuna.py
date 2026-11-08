
import csv
from pathlib import Path

import optuna
import matplotlib.pyplot as plt
from optuna.trial import TrialState


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Загружаем уже существующее исследование
study = optuna.load_study(
    study_name="lr0_search",
    storage="sqlite:///optuna.db"
)

trials = study.trials

complete = [t for t in trials if t.state == TrialState.COMPLETE]
pruned = [t for t in trials if t.state == TrialState.PRUNED]
failed = [t for t in trials if t.state == TrialState.FAIL]

print("СТАТИСТИКА OPTUNA")
print("------------------------")
print("Всего испытаний:", len(trials))
print("Завершено:", len(complete))
print("Остановлено досрочно:", len(pruned))
print("Ошибки:", len(failed))

# Подсчёт выполненных эпох
executed_epochs = 0
possible_epochs = 0

for trial in complete + pruned:
    epochs = trial.user_attrs.get("executed_epochs")

    if epochs is None:
        epochs = 30 if trial.state == TrialState.COMPLETE else 0

    executed_epochs += epochs
    possible_epochs += 30

saved_epochs = possible_epochs - executed_epochs

print("\nЭПОХИ")
print("------------------------")
print("Выполнено эпох:", executed_epochs)
print("Максимально возможное количество:", possible_epochs)
print("Сэкономлено эпох:", saved_epochs)

if possible_epochs > 0:
    saving_percent = saved_epochs / possible_epochs * 100
    print(f"Экономия эпох: {saving_percent:.2f}%")

print("\nЛУЧШИЙ РЕЗУЛЬТАТ")
print("------------------------")
print("Val loss:", study.best_value)
print("Параметры:", study.best_params)


# 1. График истории оптимизации
valid_trials = [
    t for t in complete
    if t.value is not None
]

valid_trials.sort(key=lambda t: t.number)

numbers = [t.number for t in valid_trials]
values = [t.value for t in valid_trials]

best_values = []
current_best = float("inf")

for value in values:
    current_best = min(current_best, value)
    best_values.append(current_best)

plt.figure(figsize=(10, 5))

plt.scatter(numbers, values, label="Val loss испытания")
plt.plot(numbers, best_values, label="Лучшее значение")

plt.xlabel("Номер испытания")
plt.ylabel("Validation loss")
plt.title("История оптимизации Optuna")
plt.legend()
plt.grid()

plt.savefig(
    REPORTS_DIR / "optimization_history.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# 2. График важности гиперпараметров
if len(complete) >= 2:
    try:
        importances = optuna.importance.get_param_importances(study)

        names = list(importances.keys())
        scores = list(importances.values())

        plt.figure(figsize=(9, 5))

        plt.barh(names, scores)
        plt.xlabel("Важность")
        plt.title("Важность гиперпараметров Optuna")
        plt.gca().invert_yaxis()
        plt.grid(axis="x")

        plt.savefig(
            REPORTS_DIR / "parameter_importance.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print("\nВажность гиперпараметров:")
        for name, score in importances.items():
            print(f"{name}: {score:.4f}")

    except Exception as error:
        print("Не удалось рассчитать важность:", error)


# 3. Сохраняем результаты всех испытаний в CSV
with open(
    REPORTS_DIR / "optuna_trials.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "trial",
        "state",
        "val_loss",
        "executed_epochs",
        "lr",
        "n_layers",
        "hidden",
        "dropout",
        "weight_decay",
        "batch_size"
    ])

    for trial in trials:
        writer.writerow([
            trial.number,
            trial.state.name,
            trial.value,
            trial.user_attrs.get("executed_epochs"),
            trial.params.get("lr"),
            trial.params.get("n_layers"),
            trial.params.get("hidden"),
            trial.params.get("dropout"),
            trial.params.get("weight_decay"),
            trial.params.get("batch_size")
        ])

print("\nФайлы сохранены в папке reports")
