from PIL import Image
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
# import filetype

import os

__all__ = ['convert_image','convert_image_folder' ]

def convert_image(input_path, output_path, newsize = None):
    """
    Функция для конвертации изображения в формат JPG
    
    Параметры:
    input_path (str): путь к исходному файлу
    output_path (str): путь для сохранения конвертированного файла
    newsize (int,int): если не None, то ресайзит с сохр. пропорций
    Возвращает:
    bool: True если конвертация прошла успешно, False в случае ошибки
    """
    try:
        # Проверяем существование исходного файла
        if not os.path.exists(input_path):
            print(f"Файл не найден: {input_path}")
            return False
        
        # Открываем изображение
        image = Image.open(input_path)
        
        # Создаем директорию для сохранения, если она не существует
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        w,h = image.size
        if newsize !=None:
            image.thumbnail(newsize, Image.Resampling.LANCZOS)
        # Сохраняем в формате JPG
        image.save(output_path, 'JPEG', quality=95)
        # print(image.size)

        
        return True
    
    except Exception as e:
        print(f"Произошла ошибка при конвертации: {str(e)}")
        return False

def convert_image_folder(img_folder_src_path, img_folder_dsc_path = None,  size = (416,416) ):
    """
    Функция для конвертации всех изображений в указанной папке в заданный размер.

    Параметры:
    img_folder_src_path (str): путь к исходной папке с изображениями
    img_folder_dsc_path (str, optional): путь к папке для сохранения конвертированных изображений. 
                                        Если не указан, используется та же папка, что и исходная
    size (tuple, optional): кортеж с размерами (ширина, высота) для конвертации изображений. 
                           По умолчанию (416, 416)

    Функция обрабатывает следующие форматы изображений:
    ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    Пример использования:
    convert_image_folder('path/to/images', 'path/to/converted_images', (256, 256))
    """
    if img_folder_dsc_path is None:
        img_folder_dsc_path = img_folder_src_path
    
    for img_src, img_dsc in zip(Path(img_folder_src_path).iterdir(), Path(img_folder_dsc_path).iterdir()):
        # if filetype.is_image(img):
        if str(img).lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            convert_image(img,img,size)