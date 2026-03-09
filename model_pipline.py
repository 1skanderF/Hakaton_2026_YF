from ultralytics import YOLO
import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PartResult:
    """Результат для одной части растения"""
    name: str  # лист/корень/стебель
    length_cm: float
    area_cm2: float
    confidence: float
    mask_path: str

class PlantAnalyzer:
    def __init__(self,type_model, px_per_cm: float = 93):
        models=["models/argula_best.pt","models/wheat_best.pt"]
        self.type_model=type_model
        self.model = YOLO(models[self.type_model])
        self.px_per_cm = px_per_cm
        self.class_names = {0: "лист", 1: "корень", 2: "стебель"}
        self.class_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}  # BGR
        os.makedirs("masks", exist_ok=True)
        os.makedirs("segmented", exist_ok=True)
    
    def _skeletonize(self, mask):
        skel = np.zeros(mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        done = False
        while not done:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skel = cv2.bitwise_or(skel, temp)
            mask = eroded.copy()
            done = cv2.countNonZero(mask) == 0
        return skel
    
    def create_segmented_image(self, image_path: str, masks_data: List, classes: List) -> str:
        """Создает изображение с наложенными масками без боксов"""
        img = cv2.imread(image_path)
        H, W = img.shape[:2]
        
        # Создаем маску для всех объектов
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        
        for i, mask in enumerate(masks_data):
            mask_orig = cv2.resize(mask.astype(np.uint8), (W, H), 
                                  interpolation=cv2.INTER_NEAREST)
            class_id = int(classes[i])
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # Добавляем в общую маску
            combined_mask = cv2.bitwise_or(combined_mask, mask_orig)
            
            # Окрашиваем маску в цвет класса
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask_orig > 0, color[c], colored_mask[:, :, c])
        
        # Накладываем маску на оригинал с прозрачностью
        alpha = 0.5
        result = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
        
        # Добавляем контуры для наглядности
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
        
        # Сохраняем результат
        output_path = f"segmented/segmented_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result)
        
        return output_path
    
    def analyze(self, image_path: str,) -> tuple[List[PartResult], Optional[str]]:
        """Анализ изображения и возврат результатов + пути к сегментированному изображению"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Файл {image_path} не найден")
        
        H, W = img.shape[:2]
        conf=[0.05,0.7]
        results = self.model(image_path, conf=conf[self.type_model])[0]
        
        if results.masks is None:
            return [], None
        
        masks = (results.masks.data * 255).cpu().numpy()
        classes = results.boxes.cls
        confs = results.boxes.conf
        
        parts = []
        masks_list = []
        classes_list = []
        
        for i, mask in enumerate(masks):
            mask_orig = cv2.resize(mask.astype(np.uint8), (W, H), 
                                  interpolation=cv2.INTER_NEAREST)
            
            # Измерения
            length_px = cv2.countNonZero(self._skeletonize(mask_orig.copy()))
            length_cm = length_px / self.px_per_cm
            area_cm2 = cv2.countNonZero(mask_orig) / (self.px_per_cm ** 2)
            
            # Сохраняем отдельную маску
            class_name = self.class_names.get(int(classes[i]), "unknown")
            mask_path = f"masks/{class_name}_{i}.png"
            cv2.imwrite(mask_path, mask_orig)
            
            parts.append(PartResult(
                name=class_name,
                length_cm=length_cm,
                area_cm2=area_cm2,
                confidence=float(confs[i]),
                mask_path=mask_path
            ))
            
            masks_list.append(mask)
            classes_list.append(classes[i])
        
        # Создаем сегментированное изображение
        segmented_path = self.create_segmented_image(image_path, masks_list, classes_list)
        
        return parts, segmented_path

def format_for_telegram(parts: List[PartResult]) -> str:
    """Форматирование результатов для отправки в Telegram"""
    if not parts:
        return "Ничего не найдено на изображении"
    
    # Группируем по типам
    stats = {}
    for p in parts:
        if p.name not in stats:
            stats[p.name] = {'count': 0, 'length': 0, 'area': 0}
        stats[p.name]['count'] += 1
        stats[p.name]['length'] += p.length_cm
        stats[p.name]['area'] += p.area_cm2
    
    text = "🌱 Результаты анализа:\n\n"
    
    # Детально по каждой части
    for i, p in enumerate(parts, 1):
        text += f"{i}. {p.name.capitalize()}\n"
        text += f"Длина: {p.length_cm:.1f} см\n"
        text += f"Площадь: {p.area_cm2:.1f} см²\n"
        text += f"Точность: {p.confidence:.0%}\n\n"
    
    # Общая статистика
    text += "Сводка:\n"
    for name, data in stats.items():
        text += f"   {name.capitalize()}: {data['count']} шт.\n"
        text += f"   Общая длина: {data['length']:.1f} см\n"
        text += f"   Общая площадь: {data['area']:.1f} см²\n\n"
    
    return text


import matplotlib
matplotlib.use('Agg')  # Использовать без GUI
import matplotlib.pyplot as plt


if __name__ == "__main__":
    image_path = r"D:\Hakaton_2026_YF\УФИЦРАН19022026\wheat\wheat_20260219135959626.jpg"
    
    analyzer = PlantAnalyzer(1)
    results, segmented_img = analyzer.analyze(image_path)

    print(f"Найдено частей: {len(results)}")
    for i, part in enumerate(results):
        print(f"{i+1}. {part.name}:")
        print(f"   Длина: {part.length_cm:.1f} см")
        print(f"   Площадь: {part.area_cm2:.1f} см²")

    # Сохраняем сравнение в файл вместо показа
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    segmented = cv2.imread(segmented_img)
    segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Оригинал')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title('Сегментированное')
    plt.axis('off')
    
    # Сохраняем в файл
    output_path = "comparison_result.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n✅ Сравнение сохранено: {output_path}")
    
    # Или просто смотрим что файлы создались
    print(f"✅ Сегментированное изображение: {segmented_img}")
    print(f"✅ Маски сохранены в папке: masks/")