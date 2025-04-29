import ssl # решение проблем со скачиванием файлов на некоторых устройствах 
ssl._create_default_https_context = ssl._create_stdlib_context

import os # работа с файловой системой
import sys # работа с операционной системой
import time # получение значений текущего времени
import copy # библиотека для копирования переменных
import psutil # функции для работы с процессами операционной системы
import random # генератор случайных чисел
import platform # информация о платформе
import numpy as np # основные вычисления 
import pandas as pd # работа с набором данных
from PIL import Image # загрузка изображений
from pathlib import Path # работа с путями к файлам
import matplotlib.pyplot as plt # визуализация данных

import pynvml # функции для работы с памятью
import requests # запросы в сеть
import importlib # функции импорта библиотек
import subprocess # вызов некоторых процессов операционной системы
import urllib.request  # работа с адреасми в сети интернет
from tqdm.notebook import tqdm, trange # визуализация процесса работы

import torch # PyTorch 
from torch import nn # Модуль для работы с нейронными сетями
import torch.utils.data # работа с данными 
import torch.optim as optim # работа с оптимизацией (обучением)
import torch.utils.data as data # работа с данными 
import torch.nn.functional as F # Модуль для работы с нейронными сетями в функциональном стиле

import torchvision # Модуль для работы с изображениями и нейронными сетями для них
from torchvision import transforms, datasets  # работа с изображениями 
from torchvision.datasets.utils import download_and_extract_archive # работа наборами данных

import timm # библиотека с большим количеством предобученных моделей нейронных сетей
from torchinfo import summary# библиотека для представления структуры нейронных сетей

# функции для работы с виртуальной машиной Python
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import gc # сбор "мусора"
from typing import Tuple, Optional, Union # провека типов
from torch.utils.data import DataLoader


__all__= ['computer_setup','setup_pytorch','gpu_memory_cleanup', 'split_dataset', 'batch_show', 'seed_everything',
	  'recall_score', 'precision_score', 'confiusion_matrix', 'f1_score', 'cohens_kappa', 'install_libraries',
          'configure_training_acceleration', 'load_timm_model', 'VanilaClassifier', 'plot_metrics']


def install_libraries(libraries):
    """
    Функция проверяет наличие библиотек и устанавливает отсутствующие.
    
    Параметры:
    libraries (list): список библиотек для проверки и установки
    
    Пример использования:
    install_libraries(['numpy', 'pandas', 'matplotlib'])
    """
    for library in libraries:
        try:
            # Пытаемся импортировать библиотеку
            importlib.import_module(library)
            print(f"Библиотека {library} уже установлена")
        except ImportError:
            print(f"Библиотека {library} не найдена, начинается установка...")
            try:
                # Используем subprocess для запуска pip
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])
                print(f"Библиотека {library} успешно установлена")
            except Exception as e:
                print(f"Ошибка при установке {library}: {str(e)}")

def configure_training_acceleration(device='cuda', 
                                  use_tf32=True,
                                  use_cudnn_benchmark=True,
                                  use_deterministic=False,
                                  use_memory_efficient=True):
    """
    Настраивает параметры ускорения обучения нейронной сети.
    
    Параметры:
        device (str): Устройство для обучения ('cuda' или 'cpu')
        use_tf32 (bool): Использовать TF32 для ускорения вычислений
        use_cudnn_benchmark (bool): Включить автоматический выбор быстрых алгоритмов CUDNN
        use_deterministic (bool): Использовать детерминированные алгоритмы (меньшая производительность)
        use_memory_efficient (bool): Оптимизировать использование памяти
        
    Возвращает:
        dict: Словарь с примененными настройками
    """
    config = {
        'device': device,
        'tf32_enabled': False,
        'cudnn_benchmark': False,
        'deterministic': False,
        'memory_efficient': False
    }
    
    if device == 'cuda' and torch.cuda.is_available():
        # Настройки TF32 (Ampere+ GPUs)
        if use_tf32 and torch.cuda.get_device_capability()[0] >= 8:
            # Включаем TF32 для операций CUDNN (ускорение)
            torch.backends.cudnn.allow_tf32 = True
            # Отключаем TF32 для матричных умножений (точность)
            torch.backends.cuda.matmul.allow_tf32 = False
            config['tf32_enabled'] = True
        
        # Автоматический выбор быстрых алгоритмов CUDNN
        torch.backends.cudnn.benchmark = use_cudnn_benchmark
        config['cudnn_benchmark'] = use_cudnn_benchmark
        
        # Детерминированные алгоритмы (меньшая производительность)
        torch.backends.cudnn.deterministic = use_deterministic
        config['deterministic'] = use_deterministic
        
        # Оптимизация использования памяти
        if use_memory_efficient:
            torch.backends.cuda.enable_flash_sdp(True)  # Включение Flash Attention если доступно
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            config['memory_efficient'] = True
    
    # Оптимизация памяти для CPU (если используется)
    if device == 'cpu' and use_memory_efficient:
        torch.set_num_threads(min(4, os.cpu_count()))
        torch.set_flush_denormal(True)
        config['memory_efficient'] = True
    
    return config

def computer_setup(verbose=True):
    """
    Анализирует аппаратные характеристики компьютера и настройки PyTorch.
    
    Параметры:
    ----------
    verbose : bool
        Если True, выводит подробную информацию в консоль.
     
    Возвращает:
    --------
    Dict[str, Any]
        Словарь с ключевыми параметрами системы:
        - cpu: информация о процессоре
        - ram: объем оперативной памяти (ГБ)
        - gpus: список GPU (если доступны)
        - torch: версия PyTorch и CUDA
        - python: версия Python
        - os: операционная система
    """
    def format_bytes(size: int) -> str:
        """Конвертирует байты в читаемый формат (GB/MB)."""
        for unit in ['GB', 'MB']:
            if size >= 1024**3:
                return f"{size / (1024**3):.2f} {unit}"
            size /= 1024
        return f"{size:.2f} MB"

    # Сбор общей информации
    system_info = {
        "cpu": {
            "name": platform.processor(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        },
        "ram": format_bytes(psutil.virtual_memory().total),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "mps": torch.backends.mps.is_available(),
        },
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
    }

    # Детальная информация о GPU
    gpus = []
    
    # Проверка NVIDIA GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "type": "NVIDIA GPU",
                "id": i,
                "name": props.name,
                "memory_total": format_bytes(props.total_memory),
                "memory_free": format_bytes(torch.cuda.mem_get_info(i)[0]),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_gpu": torch.cuda.device_count() > 1,
            })

    # Проверка Apple M1/M2 GPU
    elif torch.backends.mps.is_available():
        gpus.append({
            "type": "Apple MPS",
            "name": "M1/M2 GPU",
            "memory_total": "Shared with RAM",  # У Apple память унифицирована
        })

    system_info["accelerator"] = gpus

    if torch.cuda.is_available():
        system_info["cuda"] = {
            "version": torch.version.cuda,
            "cuDNN": torch.backends.cudnn.version(),
            "fp16_support": torch.cuda.get_device_capability(0)[0] >= 7
        }


    # Вывод информации в консоль
    if verbose:
        print("\n=== System Configuration ===")
        print(f"OS: {system_info['os']}")
        print(f"Python: {system_info['python']}")
        print(f"PyTorch: {system_info['torch']['version']}")
        if system_info['torch']['cuda']:
            print(f"CUDA: {system_info['torch']['cuda']}")
        print(f"\nCPU: {system_info['cpu']['name']}")
        print(f"Cores (physical/logical): {system_info['cpu']['cores_physical']}/{system_info['cpu']['cores_logical']}")
        print(f"RAM: {system_info['ram']}")

        if gpus:
            print("\n=== Accelerator Details ===")
            
            print(f"\nNumber of acc: {len(gpus)}")
            for gpu in gpus:
                print(f"\nGPU Type: {gpu['type']}")
                if gpu['type'] == "NVIDIA GPU":
                    print(f"Name: {gpu['name']} (ID: {gpu['id']})")
                    print(f"Memory: Total={gpu['memory_total']}, Free={gpu['memory_free']}")
                    print(f"Compute Capability: {gpu['compute_capability']}")

                elif gpu['type'] == "Apple MPS":
                    print(f"Name: {gpu['name']}")
                    print("Memory: Shared with system RAM")
                
            if len(gpus) > 1:
                print(f"\nTotal GPUs: {len(gpus)} (Multi-GPU setup)")

            
            if gpu['type'] == "NVIDIA GPU":
                print(f"cuda version: {system_info['cuda']['version']}")
                print(f"cudnn version: {system_info['cuda']['cuDNN']}")
                print(f"fp16 support: {system_info['cuda']['fp16_support']}")

        else:
            print("\nNo GPU acceleration available")

        print("=" * 30)

    return system_info



def setup_pytorch(
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    deterministic: bool = False,
    verbose: bool = True,
) :
    """
    Настраивает PyTorch для экспериментов (устройство, тип данных, воспроизводимость).

    Параметры:
    ----------
    device : str
        "auto" - автоматический выбор (MPS > CUDA > CPU),
        "mps" - Apple M1/M2 GPU,
        "cuda" - NVIDIA GPU,
        "cpu" - CPU.
    dtype : torch.dtype
        Желаемый тип данных (по умолчанию torch.float32).
    seed : int
        Случайный seed для воспроизводимости (по умолчанию 42).
    deterministic : bool
        Если True, включает детерминированные алгоритмы (может снижать производительность).
    verbose : bool
        Если True, выводит информацию о настройках.

    Возвращает:
    -----------
    torch.device
        Устройство, на котором будет выполняться вычисления.
    torch.int
        Потенциально доступное число процессов для работы с данными
    Пример:
    -------
    >>> device = setup_pytorch(device="auto", dtype=torch.float32, seed=42)
    """
    # Установка типа данных
    torch.set_default_dtype(dtype)

    # Определение устройства
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"  # Apple M1/M2 GPU
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        else:
            device = "cpu"
    device = torch.device(device)

    # Фиксация seed для воспроизводимости
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif device.type == "mps":
            # На M1/M2 пока нет аналога manual_seed_all, но можно использовать общий seed
            pass

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    
    # Функция для определения оптимального числа workers
    
    if device.type == 'cuda':
        n_workers = min(4, torch.cuda.device_count() * 4)  # 4 workers на GPU
    elif device.type == 'mps':
        n_workers = 2  # MPS обычно лучше с меньшим числом workers
    else:
        n_workers = min(4, os.cpu_count() // 2)  # Для CPU
    
    # Вывод информации
    if verbose:
        print("PyTorch setup:")
        print(f"- Device: {device}")
        print(f"- Default dtype: {dtype}")
        print(f"- Seed: {seed}")
        print(f"- Deterministic: {deterministic}")
        print(f"- Aviliable workers: {n_workers}")
        if device.type == "cuda":
            print(f"- CUDA device: {torch.cuda.get_device_name(0)}")
        elif device.type == "mps":
            print("- Apple M1/M2 GPU (MPS) enabled")

    return device, n_workers




def gpu_memory_cleanup(device = None, verbose = True):
    """
    Проверяет статистику памяти GPU и очищает кэш PyTorch.
    Поддерживает NVIDIA (CUDA) и Apple (MPS) GPU.

    Параметры:
    ----------
    device : torch.device, optional
        Устройство для проверки (по умолчанию текущее устройство PyTorch)
    verbose : bool
        Если True, печатает детальную информацию

    Возвращает:
    --------
    dict
        Словарь с информацией о памяти:
        {
            'memory_allocated': int,  # Текущее использование памяти PyTorch
            'memory_reserved': int,   # Зарезервированная память PyTorch
            'memory_free': int,      # Свободная память на устройстве
            'memory_total': int      # Общая память на устройстве
        }
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 'cpu')

    memory_stats = {}

    try:
        # Для NVIDIA GPU
        if device.type == 'cuda':
            # Получаем статистику через PyTorch
            memory_stats['memory_allocated'] = torch.cuda.memory_allocated(device)
            memory_stats['memory_reserved'] = torch.cuda.memory_reserved(device)
            
            # Получаем общую статистику через NVML
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device.index or 0)
            info = nvmlDeviceGetMemoryInfo(handle)
            memory_stats['memory_free'] = info.free
            memory_stats['memory_total'] = info.total
            
            if verbose:
                print(f"\n=== NVIDIA GPU Memory ===")
                print(f"Device: {torch.cuda.get_device_name(device)}")
                print(f"Memory allocated by PyTorch: {memory_stats['memory_allocated'] / 1024**2:.2f} MB")
                print(f"Memory reserved by PyTorch: {memory_stats['memory_reserved'] / 1024**2:.2f} MB")
                print(f"Free memory on GPU: {memory_stats['memory_free'] / 1024**2:.2f} MB")
                print(f"Total GPU memory: {memory_stats['memory_total'] / 1024**3:.2f} GB")

        # Для Apple M1/M2 GPU
        elif device.type == 'mps':
            # MPS пока не предоставляет детальной статистики памяти
            memory_stats['memory_allocated'] = -1
            memory_stats['memory_reserved'] = -1
            memory_stats['memory_free'] = -1
            memory_stats['memory_total'] = -1
            
            if verbose:
                print("\n=== Apple MPS GPU ===")
                print("Memory statistics not available for MPS devices")

        # Очистка памяти
        if device.type in ['cuda', 'mps']:
            if verbose:
                print("\nPerforming memory cleanup...")
            
            # Освобождаем кэш PyTorch
            torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()
            
            # Принудительный сбор мусора Python
            gc.collect()
            
            if verbose:
                print("Memory cleanup completed!")
                
    except Exception as e:
        if verbose:
            print(f"\nError during memory check: {str(e)}")
    
    return memory_stats


def split_dataset(
    dataset: torch.utils.data.Dataset,
    val_ratio: float = 0.2,
    val_transforms = None,
    device = None
) -> Tuple:
    """
    Разделяет датасет на тренировочную и валидационную части с автоматической
    настройкой для GPU (CUDA), Apple M1/M2 (MPS) или CPU.

    Параметры:
    ----------
    dataset : torch.utils.data.Dataset
        Полный датасет для разделения
    val_ratio : float, optional
        Доля данных для валидации (по умолчанию 0.2)
    val_transforms: torchvision.transform, optional
        Преобразования данных для валидационного набора данных
    device : torch.device, optional
        Устройство для размещения данных (автоопределение если None)      
    Возвращает:
    --------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader) - загрузчики для тренировочной и валидационной частей

    Пример:
    -------
    >>> train_loader, val_loader = split_dataset(full_dataset, val_ratio=0.1)
    """
    # Автоопределение устройства если не указано
    if device is None:
        # device = torch.device('cuda' if torch.cuda.is_available() else
        #                     'mps' if torch.backends.mps.is_available() else 'cpu')
        device = 'cpu'
    
    g = torch.Generator(device = device)
    
    # Размеры частей
    val_size = int(len(dataset) * val_ratio)
    
    train_size = len(dataset) - val_size

    # Разделение датасета
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size], generator=g)

    val_dataset = copy.deepcopy(val_dataset)

    if val_transforms:
        val_dataset.dataset.transform = val_transforms
    
    return train_dataset, val_dataset


def seed_everything(seed: int = 42, deterministic_only: bool = True) -> None:
    """
    Инициализирует все генераторы случайных чисел для обеспечения воспроизводимости результатов.
    
    Параметры:
    seed (int): начальное значение для генераторов случайных чисел (по умолчанию 42)
    deterministic_only (bool): если True, отключает оптимизацию cudnn для полностью детерминированного поведения (по умолчанию True)
    
    Примечания:
    - Установка PYTHONHASHSEED необходима для детерминированного поведения словаря
    - manual_seed устанавливает seed для CPU и GPU
    - cuda.manual_seed устанавливает seed только для GPU
    - cudnn.deterministic обеспечивает детерминированное поведение сверток
    - cudnn.benchmark отключает оптимизацию для детерминированного поведения
    """
    
    # Проверяем тип входных данных
    if not isinstance(seed, int):
        raise TypeError("seed должен быть целочисленным значением")
    
    # Инициализация стандартных генераторов
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Инициализация PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # для всех GPU
    
    # Настройка детерминированного поведения
    if deterministic_only:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def batch_show(
    images: torch.Tensor,  # входные изображения в формате torch.Tensor
    nrow: int = 4,         # количество изображений в ряду
    figsize: tuple = (8, 8),  # размер фигуры в дюймах (ширина, высота)
    mean: float = 0.0,     # среднее значение для денормализации
    std: float = 1.0       # стандартное отклонение для денормализации
):
    """
    Функция для отображения сетки изображений с возможностью денормализации.
    
    Параметры:
    - images: torch.Tensor - тензор изображений [batch_size, channels, height, width]
    - nrow: int - количество изображений в ряду
    - figsize: tuple - кортеж с размерами фигуры (ширина, высота)
    - mean: float - среднее значение для денормализации
    - std: float - стандартное отклонение для денормализации
    
    Возвращаемое значение:
    None - отображает изображения
    """
    try:
        # Создаем сетку изображений
        img = torchvision.utils.make_grid(images, nrow=nrow, padding=0)
        
        # Транспонируем оси для корректного отображения
        img = img.transpose(2, 0).transpose(0, 1)
        
        # Преобразуем параметры денормализации в тензоры
        std_tensor = torch.as_tensor(std)
        mean_tensor = torch.as_tensor(mean)
        
        # Денормализуем изображение
        img = (img * std_tensor + mean_tensor)
        
        # Преобразуем в numpy массив
        img = img.data.cpu().numpy()
        
        # Создаем фигуру и отображаем изображение
        plt.figure(figsize=figsize)
        plt.imshow(img, interpolation='lanczos')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при отображении изображений: {e}")



def recall_score(y_pred, y_true, average='macro'):
    """
    Вычисляет Recall для многоклассовой классификации.
    
    Args:
        y_pred: Предсказанные вероятности (logits) [batch_size, num_classes]
        y_true: Истинные метки [batch_size]
        average: 'macro' (усреднение по классам), 'micro' (глобальный)
    """
    num_classes = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, dim=1)
    
    recall_per_class = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()

        recall = tp / (tp + fn + 1e-6)  # Добавляем 1e-6 для избежания деления на 0
        recall_per_class.append(recall)
    
    if average == 'macro':
        return torch.mean(torch.stack(recall_per_class))
    elif average == 'micro':
        tp_total = sum(((y_pred == c) & (y_true == c)).sum() for c in range(num_classes))
        fn_total = sum(((y_pred != c) & (y_true == c)).sum() for c in range(num_classes))
        return tp_total / (tp_total + fn_total + 1e-6)
    else:
        raise ValueError("average должен быть 'macro' или 'micro'")

def precision_score(y_pred, y_true, average='macro'):
    """
    Вычисляет Precision для многоклассовой классификации.
    
    Args:
        y_pred: Предсказанные вероятности (logits) [batch_size, num_classes]
        y_true: Истинные метки [batch_size]
        average: 'macro' (усреднение по классам), 'micro' (глобальный)
    """
    num_classes = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, dim=1)
    
    precision_per_class = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()

        precision = tp / (tp + fp + 1e-6)  # Добавляем 1e-6 для избежания деления на 0
        precision_per_class.append(precision)
    
    if average == 'macro':
        return torch.mean(torch.stack(precision_per_class))
    elif average == 'micro':
        tp_total = sum(((y_pred == c) & (y_true == c)).sum() for c in range(num_classes))
        fp_total = sum(((y_pred == c) & (y_true != c)).sum() for c in range(num_classes))
        return tp_total / (tp_total + fp_total + 1e-6)
    else:
        raise ValueError("average должен быть 'macro' или 'micro'")
 

def confiusion_matrix(predictions, targets, num_classes):
    """
    Рассчитывает confiusion_matrix для multiclass классификации
    
    Параметры:
    predictions (torch.Tensor): тензор с предсказаниями (batch_size)
    targets (torch.Tensor): тензор с истинными метками (batch_size)
    num_classes (int): количество классов
    
    Возвращает:
    precision (torch.Tensor): тензор с precision для каждого класса
    """
    
    # Проверяем размерности
    assert predictions.size() == targets.size(), "Размерности предсказаний и таргетов должны совпадать"
    
    # Создаем confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    # Заполняем confusion matrix
    for p, t in zip(predictions.view(-1), targets.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
      
    return conf_matrix

            
def f1_score(y_pred, y_true, average='macro'):
        """
        Вычисляет F1-score (гармоническое среднее Precision и Recall).
        """
        precision = precision_score(y_pred, y_true, average)
        recall    = recall_score(y_pred, y_true, average)
        return 2 * (precision * recall) / (precision + recall + 1e-6)

def cohens_kappa(y_pred, y_true, num_classes):
    """
    Вычисляет Cohen's Kappa для многоклассовой классификации.
    
    Args:
        y_pred: Предсказанные вероятности (logits) [batch_size, num_classes]
        y_true: Истинные метки [batch_size]
        num_classes: Количество классов
    """
    y_pred = torch.argmax(y_pred, dim=1)
    
    # Матрица ошибок (confusion matrix)
    conf_matrix = torch.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        conf_matrix[t.long(), p.long()] += 1
    
    # Наблюдаемая точность (p_o)
    p_o = torch.trace(conf_matrix) / torch.sum(conf_matrix)
    
    # Ожидаемая точность (p_e)
    row_sums = torch.sum(conf_matrix, dim=1)
    col_sums = torch.sum(conf_matrix, dim=0)
    p_e = torch.sum(row_sums * col_sums) / (torch.sum(conf_matrix) ** 2)
    
    kappa = (p_o - p_e) / (1 - p_e + 1e-6)  # Добавляем 1e-6 для стабильности
    return kappa

def plot_metrics(df: pd.DataFrame, metrics: list, figsize: tuple =(3, 3), save_path: str = None):
        """
        Функция для построения графиков обучения модели
        
        Параметры:
        df (pd.DataFrame) - входной DataFrame с данными
        metrics (list) - список метрик для построения графиков
        figsize (tuple) - Размер фигуры
        save_path (str) - путь для сохранения графиков (опционально)
        """
        
        # Проверяем наличие необходимых столбцов
        required_columns = {'epoch', 'val_loss', 'train_loss'}
        if not required_columns.issubset(df.columns):
            raise ValueError("DataFrame должен содержать столбцы: epoch, val_loss, train_loss")
            
        # Проверяем наличие всех метрик в DataFrame
        if not set(metrics).issubset(df.columns):
            raise ValueError("Некоторые метрики отсутствуют в DataFrame")
    
        # Настройка стиля графиков        
        plt.figure(figsize=figsize)
        
        # График потерь
        plt.subplot(2, 1, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Графики метрик
        plt.subplot(2, 1, 2)
        for metric in metrics:
            plt.plot(df['epoch'], df[metric], label=metric)
        plt.title('Metrics Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        # Настройка общего вида
        plt.tight_layout()
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path, dpi=100)
            
        plt.show()
   	

def load_timm_model(model_name: str, num_classes: int, freeze_encoder: bool = True, drop_rate: float = 0.0):
    """
    Загружает модель TIMM с замороженным кодировщиком и заданным числом выходных классов.
    
    Параметры:
    - model_name: название модели TIMM
    - num_classes: количество выходных классов
    - freeze_encoder: флаг замораживания кодировщика (по умолчанию True)
    - drop_rate: регуляризация выходного слоя
    Возвращает:
    - torch.nn.Module: загруженная и модифицированная модель
    """
    # Создаем базовую модель
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,  # Добавляем дропаут для регуляризации
    )
    
    # Замораживаем параметры кодировщика, если нужно
    if freeze_encoder:
        for param in model.parameters():
            param.requires_grad = False
            
        # Размораживаем голову модели для обучения
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        
    # Переводим модель в режим обучения

    return model



class VanilaClassifier:
    def __init__(self, model, optimizer, criterion, metrics=None, device='cpu', scheduler=None):
        """
        Инициализация упрощенной процедуры обучения нейронной сети.
        
        Args:
            model: Модель для обучения
            optimizer: Оптимизатор
            criterion: Функция потерь
            metrics (dict): Словарь метрик {'name': metric_fn}
            device: Устройство для обучения ('cpu' или 'cuda')
            scheduler: Планировщик скорости обучения
        """
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.metrics = metrics if metrics is not None else {}        
        self.scheduler = scheduler
        self.history = None

          
    @staticmethod
    def epoch_time(start_time, end_time):
        """Вычисление времени эпохи."""
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def train_epoch(self, dataloader):
        """Обучение модели на одной эпохе."""
        epoch_loss = 0
        train_metrics = {name: 0 for name in self.metrics.keys()}
        
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for x, y in tqdm(dataloader, desc="Training", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            for name, metric in self.metrics.items():
                train_metrics[name] += metric(y_pred, y).item()
                
        for name in train_metrics:
            train_metrics[name] /= len(dataloader)
            
        return epoch_loss / len(dataloader), train_metrics
    
    def evaluate_epoch(self, dataloader):
        """Оценка модели на одной эпохе."""
        epoch_loss = 0
        val_metrics = {name: 0 for name in self.metrics.keys()}
        
        self.model.eval()
        
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                
                y_pred = self.model(x)
                epoch_loss += self.criterion(y_pred, y).item()
                
                for name, metric in self.metrics.items():
                    val_metrics[name] += metric(y_pred, y).item()
                    
        for name in val_metrics:
            val_metrics[name] /= len(dataloader)
            
        return epoch_loss / len(dataloader), val_metrics
    
    def fit(self, 
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader, 
            epochs: int = 10, 
            path_best: str = 'best_model.pt', 
            verbose: int = 1) -> pd.DataFrame:
        """
        Метод для обучения модели на данных.
    
        Параметры:
        train_loader (DataLoader): загрузчик данных для обучения
        val_loader (DataLoader): загрузчик данных для валидации
        epochs (int): количество эпох обучения (по умолчанию 10)
        path_best (str): путь для сохранения лучшей модели (по умолчанию 'best_model.pt')
        verbose (int): частота вывода логов (по умолчанию 1 - вывод после каждой эпохи), 
                        если <1 -не выводит логи
    
        Возвращает:
        pd.DataFrame: DataFrame с историей обучения, содержащий:
            - train_loss: потери на обучающей выборке
            - val_loss: потери на валидационной выборке
            - метрики (если заданы в self.metrics)
    
        Процесс обучения включает:
        1. Инициализацию истории обучения
        2. Проход по всем эпохам
        3. Обучение на тренировочных данных
        4. Валидацию на валидационных данных
        5. Обновление планировщика (если есть)
        6. Сохранение лучшей модели
        7. Логирование результатов
        """
        self.history = {'epoch':[],'LR':[],
                        'train_loss': [], 'val_loss': [], }
        if self.metrics:
            for name in self.metrics.keys():
                self.history[f'train_{name}'] = []
                self.history[f'val_{name}'] = []
        best_val_loss = float('inf')
        
        for epoch in trange(epochs):
            start_time = time.monotonic()
            
            # Обучение и валидация
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate_epoch(val_loader)
            
            # Обновление планировщика
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
            # Сохранение лучшей модели
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), path_best)
                
            # Запись истории
            self.history['epoch'].append(epoch)
            self.history['LR'].append(self.optimizer.param_groups[0]["lr"])
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if self.metrics:
                for name in self.metrics.keys():
                    self.history[f'train_{name}'].append(train_metrics[name])
                    self.history[f'val_{name}'].append(val_metrics[name])
                    
            # Логирование
            end_time = time.monotonic()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | LR {self.optimizer.param_groups[0]["lr"]:.4f}')
                print(f"Train: Loss: {train_loss:.2f}", end='')
                if self.metrics:
                    for name in self.metrics.keys():
                        print(f" | {name.capitalize()}: {train_metrics[name]:.2f}", end='')
                print(f"\nVal:   Loss: {val_loss:.2f}", end='')
                if self.metrics:
                    for name in self.metrics.keys():
                        print(f" | {name.capitalize()}: {val_metrics[name]:.2f}", end='')
                print('\n')
                
        return pd.DataFrame(self.history)

    def predict(self,
            data,
            return_probs = False,
            return_logits= False):
        """
        Метод для предсказания.
        	
        Параметры:
        - data: входные данные (могут быть torch.Tensor, np.ndarray, DataLoader или Dataset)
        - return_probs: возвращать вероятности (softmax)
        - return_logits: возвращать логиты
        
        Возвращает:
        - предсказания (и вероятности/логиты, если указано)
        """
        self.model.eval()
        
        # Конвертация входных данных
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        elif isinstance(data, torch.utils.data.Dataset):
            data = torch.utils.data.DataLoader(data, batch_size=32)
        elif isinstance(data, torch.utils.data.DataLoader):
            pass  # оставляем как есть
        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)
        else:
            raise ValueError("Неподдерживаемый тип данных")
            
        with torch.inference_mode():
            if isinstance(data, torch.utils.data.DataLoader):
                predictions = []
                probs = []
                logits = []
                
                for batch in tqdm(data, desc="Predicting", leave=False):
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]  # берем только данные, без меток
                    batch = batch.to(self.device)
                    
                    logits_batch = self.model(batch)
                    probs_batch = torch.softmax(logits_batch, dim=-1)
                    preds_batch = torch.argmax(probs_batch, dim=-1)
                    
                    predictions.append(preds_batch)
                    probs.append(probs_batch)
                    logits.append(logits_batch)
                
                predictions = torch.cat(predictions)
                probs = torch.cat(probs)
                logits = torch.cat(logits)
            else:
                logits = self.model(data)
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                
        if return_probs and return_logits:
            return predictions, probs, logits
        elif return_probs:
            return predictions, probs
        elif return_logits:
            return predictions, logits
        else:
            return predictions   

    def load_best_model(self, path='best_model.pt'):
        """Загрузка лучших весов модели из файла."""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()  # Переводим модель в режим оценки
        print(f"Модель успешно загружена из {path}")  