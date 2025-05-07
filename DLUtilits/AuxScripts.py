from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import math

__all__= ['yolo_results', 'yolo_predict_structure', 'add_ground_truth_labels', 'yolo_predict_with_gt', 'visualize_yolo_results', 'calculate_map_python']

def round_(x, digits=2):
    return (x*100//1)/100

def yolo_predict_structure(img_path, result):
    """
    Создает базовую структуру предсказаний для одного изображения
    
    Args:
        img_path: Путь к изображению
        result: Результаты детекции для этого изображения
        
    Returns:
        Словарь с базовой структурой предсказаний
    """
    img = Image.open(img_path)
    w, h = img.size
    
    output = {
        'name': img_path.stem,
        'size': [w, h],
        'predict': []
    }
    
    for box in result.boxes:
        # Координаты bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cls = int(box.cls)
        conf = round_(float(box.conf))

        output['predict'].append([cls, conf, round_(x1), round_(y1), round_(x2), round_(y2)])
    
    return output


def add_ground_truth_labels(output, labels_dir):
    """
    Добавляет ground truth разметку к структуре предсказаний
    
    Args:
        output: Базовая структура предсказаний
        labels_dir: Путь к папке с YOLO разметкой
        
    Returns:
        Структура с добавленными ground truth метками
    """
    output['labels'] = []
    label_file = Path(labels_dir) / (output['name'] + '.txt')
    
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Конвертируем YOLO формат в пиксельные координаты
                w, h = output['size']
                x_center *= w
                y_center *= h
                width *= w
                height *= h
                
                x1 = round_(x_center - width / 2)
                y1 = round_(y_center - height / 2)
                x2 = round_(x_center + width / 2)
                y2 = round_(y_center + height / 2)
                
                output['labels'].append([class_id, x1, y1, x2, y2])
    
    return output

def yolo_predict_with_gt(img_folder, labels_dir, results, n_imgs=2):
    """
    Формирование массива ответов    
    Args:
        img_folder: Путь к папке с тестовыми изображениями
        labels_dir: Путь к папке с YOLO разметкой (.txt файлы)
        results: структура результатов
        n_imgs: число изображений для проверки, если None, обрабатываются все доступные изображения.
    """
    if n_imgs is None:
        n_imgs = len(list(Path(img_folder).iterdir()))
    n_imgs = min(n_imgs, len(results), len(list(Path(img_folder).iterdir())))          
        
    img_paths = [x for x,_ in zip(Path(img_folder).iterdir(), range(n_imgs)) if str(x).endswith(('jpg','png'))]

    outputs = []
    if not img_paths:
        print("Не найдены изображения в указанной директории.")
        return outputs
    
    for img_path, result in zip(img_paths, results):
        # Создаем базовую структуру предсказаний
        output = yolo_predict_structure(img_path, result)
        # Добавляем ground truth метки
        output = add_ground_truth_labels(output, labels_dir)
        outputs.append(output)
  
    return outputs

def yolo_results(model, img_folder, n_imgs=2):
    if n_imgs is None:
        n_imgs = len(list(Path(img_folder).iterdir()))
    n_imgs = min(n_imgs, len(list(Path(img_folder).iterdir())))     
        
    img_paths = [x for x,_ in zip(Path(img_folder).iterdir(), range(n_imgs)) if str(x).endswith(('jpg','png'))]

    if not img_paths:
        print("Не найдены изображения в указанной директории.")
        return
    results = model.predict(img_paths, save=False) 
    return results

def visualize_yolo_results(outs, show_labels=True, show_predictions=Trueб figsize = (3,3)):
    """
    Визуализация разметки и предсказаний YOLO
    
    Args:
        outs: массив с данными о разметке и предсказаниях (результат работы yolo_predictions_with_gt)
        show_labels: показывать ground truth разметку
        show_predictions: показывать предсказания модели
    """
    # Цвета для разных классов (можно расширить)
    colors = ['red', 'blue', 'black', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    for out in outs:
        img_path = Path(test_img_folder) / (out['name'] + '.jpg')  # предполагаем jpg, можно добавить проверку на png
        if not img_path.exists():
            img_path = Path(test_img_folder) / (out['name'] + '.png')
            if not img_path.exists():
                print(f"Изображение {out['name']} не найдено")
                continue
        
        img = Image.open(img_path)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title(out['name'])
        ax = plt.gca()
        
        # Рисуем ground truth разметку
        if show_labels and out['labels']:
            for label in out['labels']:
                class_id, x1, y1, x2, y2 = label
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    fill=False, 
                                    color=colors[int(class_id) % len(colors)], 
                                    linewidth=2,
                                    linestyle='--')
                ax.add_patch(rect)
                plt.text(x1, y1 - 5, f'GT {int(class_id)}', 
                         color=colors[int(class_id) % len(colors)],
                         bbox=dict(facecolor='white', alpha=0.7))
        
        # Рисуем предсказания
        if show_predictions and out['predict']:
            for pred in out['predict']:
                class_id, conf, x1, y1, x2, y2 = pred
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    fill=False, 
                                    color=colors[int(class_id) % len(colors)], 
                                    linewidth=2)
                ax.add_patch(rect)
                plt.text(x1, y1 - 5, f'Pred {int(class_id)} {conf:.2f}', 
                         color=colors[int(class_id) % len(colors)],
                         bbox=dict(facecolor='white', alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def calculate_map_python(outs, iou_threshold=0.5):
    """
    Расчет mAP (mean Average Precision) для набора предсказаний и ground truth.
    Это стандартная реализация, совместимая с общепринятыми методами оценки в задачах object detection.
    
    Args:
        outs: список словарей с предсказаниями и ground truth
        iou_threshold: порог IoU для определения true positive (по умолчанию 0.5)
    
    Returns:
        tuple: (mAP, ap_per_class, precision_recall_per_class)
            - mAP: среднее AP по всем классам
            - ap_per_class: словарь {class_id: AP}
            - precision_recall_per_class: словарь {class_id: (precisions, recalls)}
    Ключевые особенности:        
        Точный расчет IoU с обработкой краевых случаев        
        11-point interpolation для расчета AP (как в PASCAL VOC)        
        Разделение по классам с индивидуальными метриками для каждого        
        Возвращает кривые Precision-Recall для визуализации
        
    Как работает алгоритм:
        Для каждого класса:        
            Сортировка предсказаний по confidence score        
            Сопоставление с ground truth через IoU        
            Расчет кумулятивных TP/FP        
            Построение кривой Precision-Recall        
            Расчет AP по 11 точкам        
            Усреднение AP по всем классам дает mAP       
    """
    def calculate_iou(box1, box2):
        """Вычисление Intersection over Union для двух bounding box'ов"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Координаты области пересечения
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        if inter_area == 0:
            return 0.0
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area

    # Собираем все предсказания и ground truth по классам
    class_predictions = defaultdict(list)
    class_ground_truth = defaultdict(list)

    for out in outs:
        img_name = out['name']
        gt_boxes = out['labels']
        pred_boxes = out['predict']
        
        # Группируем ground truth по классам
        for gt in gt_boxes:
            class_id = int(gt[0])
            box = gt[1:]
            class_ground_truth[class_id].append({
                'img_name': img_name,
                'box': box,
                'matched': False
            })
        
        # Группируем предсказания по классам
        for pred in pred_boxes:
            class_id = int(pred[0])
            confidence = pred[1]
            box = pred[2:]
            class_predictions[class_id].append({
                'img_name': img_name,
                'box': box,
                'confidence': confidence,
                'matched': False
            })

    ap_per_class = {}
    precision_recall_per_class = {}

    for class_id in class_predictions:
        # Получаем предсказания и GT для текущего класса
        preds = class_predictions[class_id]
        gts = class_ground_truth.get(class_id, [])
        
        # Сортируем предсказания по уверенности (по убыванию)
        preds_sorted = sorted(preds, key=lambda x: -x['confidence'])
        
        tp = [0] * len(preds_sorted)
        fp = [0] * len(preds_sorted)
        n_gt = len(gts)
        
        # Для каждого предсказания ищем наилучшее совпадение с GT
        for i, pred in enumerate(preds_sorted):
            best_iou = 0.0
            best_gt_idx = -1
            
            # Ищем GT на том же изображении
            img_gts = [gt for gt in gts if gt['img_name'] == pred['img_name']]
            
            for j, gt in enumerate(img_gts):
                if gt['matched']:
                    continue
                
                iou = calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Если IoU > порога, считаем true positive
            if best_iou >= iou_threshold:
                tp[i] = 1
                img_gts[best_gt_idx]['matched'] = True
            else:
                fp[i] = 1
        
        # Вычисляем кумулятивные суммы TP и FP
        tp_cumsum = []
        fp_cumsum = []
        current_tp = 0
        current_fp = 0
        
        for t, f in zip(tp, fp):
            current_tp += t
            current_fp += f
            tp_cumsum.append(current_tp)
            fp_cumsum.append(current_fp)
        
        # Вычисляем Precision и Recall
        recalls = [t / (n_gt + 1e-6) for t in tp_cumsum]
        precisions = []
        for t, f in zip(tp_cumsum, fp_cumsum):
            precisions.append(t / (t + f + 1e-6))

        
        # Добавляем точку (0, 1) для построения кривой
        precisions = [1.0] + precisions
        recalls = [0.0] + recalls
        
        # Сохраняем кривую Precision-Recall
        precision_recall_per_class[class_id] = (precisions, recalls)
        
        # Вычисляем AP (11-point interpolation)
        ap = 0.0
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            mask = [r >= t for r in recalls]
            if any(mask):
                max_precision = max(p for p, m in zip(precisions, mask) if m)
                ap += max_precision
        ap /= 11
        
        ap_per_class[class_id] = ap
    
    # Вычисляем mAP как среднее AP по всем классам
    map_score = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
    
    return map_score, ap_per_class, precision_recall_per_class