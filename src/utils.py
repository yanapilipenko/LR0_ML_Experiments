
import random
import numpy as np
import torch


def set_seed(seed: int, num_threads: int = 1):
    """
    Настройка генераторов случайных чисел
    для воспроизводимости экспериментов.
    """

    # Фиксируем генератор Python
    random.seed(seed)

    # Фиксируем генератор NumPy
    np.random.seed(seed)

    # Фиксируем генератор PyTorch
    torch.manual_seed(seed)

    # Фиксируем CUDA, если она используется
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Настраиваем детерминированные вычисления
    torch.use_deterministic_algorithms(True)

    # Настраиваем CPU
    torch.set_num_threads(num_threads)

    # Дополнительные настройки CUDA
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name: str):
    """
    Проверяет доступность выбранного устройства.
    """

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна на этом компьютере")

    return torch.device(device_name)
