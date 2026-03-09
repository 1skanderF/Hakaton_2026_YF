import telebot
from telebot import types
import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
from dataclasses import dataclass
from typing import List, Optional
from ultralytics import YOLO
import threading
import time
import shutil
import os
from dotenv import load_dotenv



# ==================== КЛАСС АНАЛИЗАТОРА (из вашего кода) ====================

@dataclass
class PartResult:
    """Результат для одной части растения"""
    name: str  # лист/корень/стебель
    length_cm: float
    area_cm2: float
    confidence: float
    mask_path: str

class PlantAnalyzer:
    def __init__(self, type_model, px_per_cm: float = 93):
        models = ["models/argula_best.pt", "models/wheat_best.pt"]
        self.type_model = type_model
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
    
    def analyze(self, image_path: str) -> tuple[List[PartResult], Optional[str]]:
        """Анализ изображения и возврат результатов + пути к сегментированному изображению"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Файл {image_path} не найден")
        
        H, W = img.shape[:2]
        conf = [0.05, 0.7]
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
        return "🌱 На изображении не найдено частей растений."
    
    # Группируем по типам
    stats = {}
    for p in parts:
        if p.name not in stats:
            stats[p.name] = {'count': 0, 'length': 0, 'area': 0}
        stats[p.name]['count'] += 1
        stats[p.name]['length'] += p.length_cm
        stats[p.name]['area'] += p.area_cm2
    
    text = "🌱 <b>Результаты анализа:</b>\n\n"
    
    # Детально по каждой части
    for i, p in enumerate(parts, 1):
        text += f"<b>{i}. {p.name.capitalize()}</b>\n"
        text += f"📏 Длина: {p.length_cm:.1f} см\n"
        text += f"📐 Площадь: {p.area_cm2:.1f} см²\n"
        text += f"🎯 Точность: {p.confidence:.0%}\n\n"
    
    # Общая статистика
    text += "📊 <b>Сводка:</b>\n"
    for name, data in stats.items():
        emoji = "🍃" if name == "лист" else "🌿" if name == "стебель" else "🌱"
        text += f"{emoji} {name.capitalize()}: {data['count']} шт.\n"
        text += f"   📏 Общая длина: {data['length']:.1f} см\n"
        text += f"   📐 Общая площадь: {data['area']:.1f} см²\n\n"
    
    return text

# ==================== ТЕЛЕГРАМ БОТ ====================

# Токен бота (получите у @BotFather)
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # <--- ЗАМЕНИТЕ НА СВОЙ ТОКЕН

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN, parse_mode='HTML')

# Хранилище состояний пользователей
user_data = {}
user_analyzers = {}

# Создаем папки для временных файлов
os.makedirs("temp_images", exist_ok=True)
os.makedirs("temp_results", exist_ok=True)

# Очистка старых временных файлов (запускается в отдельном потоке)
def cleanup_old_files():
    while True:
        try:
            now = time.time()
            for folder in ["temp_images", "temp_results", "masks", "segmented"]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        filepath = os.path.join(folder, filename)
                        if os.path.isfile(filepath):
                            # Удаляем файлы старше 1 часа
                            if os.path.getmtime(filepath) < now - 3600:
                                os.remove(filepath)
        except Exception as e:
            print(f"Ошибка очистки: {e}")
        time.sleep(1800)  # Проверка каждые 30 минут

# Запускаем очистку в фоне
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

# Команда /start
@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    welcome_text = (
        "🌱 <b>Добро пожаловать в Plant Analyzer Bot!</b>\n\n"
        "Я помогу проанализировать изображения растений и определить:\n"
        "• листья 🍃\n"
        "• корни 🌱\n"
        "• стебли 🌿\n\n"
        "Я измерю длину и площадь каждой части.\n\n"
        "Для начала выберите тип растения:"
    )
    
    # Создаем клавиатуру для выбора модели
    markup = types.InlineKeyboardMarkup(row_width=2)
    btn1 = types.InlineKeyboardButton("🌱 Руккола", callback_data="model_0")
    btn2 = types.InlineKeyboardButton("🌾 Пшеница", callback_data="model_1")
    markup.add(btn1, btn2)
    
    bot.send_message(user_id, welcome_text, reply_markup=markup)

# Обработка выбора модели
@bot.callback_query_handler(func=lambda call: call.data.startswith('model_'))
def model_selection(call):
    user_id = call.from_user.id
    model_type = int(call.data.split('_')[1])
    
    # Сохраняем выбор пользователя
    user_data[user_id] = {'model_type': model_type}
    
    # Создаем анализатор для пользователя
    try:
        user_analyzers[user_id] = PlantAnalyzer(model_type)
        model_name = "Руккола" if model_type == 0 else "Пшеница"
        
        bot.edit_message_text(
            chat_id=user_id,
            message_id=call.message.message_id,
            text=f"✅ Выбрана модель: <b>{model_name}</b>\n\n"
                 f"Теперь отправьте мне фотографию растения для анализа.",
            reply_markup=None
        )
    except Exception as e:
        bot.send_message(user_id, f"❌ Ошибка загрузки модели: {e}")

# Команда /help
@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = (
        "🤖 <b>Помощь по боту</b>\n\n"
        "<b>Доступные команды:</b>\n"
        "/start - Начать работу (выбрать модель)\n"
        "/help - Показать эту справку\n"
        "/reset - Сбросить выбор модели\n\n"
        "<b>Как пользоваться:</b>\n"
        "1. Выберите тип растения (руккола/пшеница)\n"
        "2. Отправьте фото растения\n"
        "3. Получите результаты анализа\n\n"
        "<b>Что анализируется:</b>\n"
        "• Листья 🍃 (зеленый)\n"
        "• Корни 🌱 (красный)\n"
        "• Стебли 🌿 (синий)\n\n"
        "Результаты включают длину и площадь каждой части."
    )
    bot.send_message(user_id, help_text)

# Команда /reset
@bot.message_handler(commands=['reset'])
def reset(message):
    user_id = message.from_user.id
    if user_id in user_data:
        del user_data[user_id]
    if user_id in user_analyzers:
        del user_analyzers[user_id]
    
    markup = types.InlineKeyboardMarkup(row_width=2)
    btn1 = types.InlineKeyboardButton("🌱 Руккола", callback_data="model_0")
    btn2 = types.InlineKeyboardButton("🌾 Пшеница", callback_data="model_1")
    markup.add(btn1, btn2)
    
    bot.send_message(
        user_id, 
        "🔄 Выбор модели сброшен. Пожалуйста, выберите тип растения:",
        reply_markup=markup
    )

# Обработка фотографий
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    user_id = message.from_user.id
    
    # Проверяем, выбрана ли модель
    if user_id not in user_data or 'model_type' not in user_data[user_id]:
        bot.send_message(
            user_id,
            "❌ Сначала выберите модель с помощью команды /start"
        )
        return
    
    if user_id not in user_analyzers:
        bot.send_message(
            user_id,
            "❌ Ошибка инициализации анализатора. Используйте /start"
        )
        return
    
    # Отправляем сообщение о начале обработки
    status_msg = bot.send_message(user_id, "🔄 Анализирую изображение...")
    
    try:
        # Получаем файл изображения
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Сохраняем временный файл с уникальным именем
        temp_filename = f"temp_images/{uuid.uuid4()}.jpg"
        with open(temp_filename, 'wb') as f:
            f.write(downloaded_file)
        
        # Анализируем
        analyzer = user_analyzers[user_id]
        results, segmented_img = analyzer.analyze(temp_filename)
        
        if not results:
            bot.edit_message_text(
                "❌ На изображении не найдено частей растений.",
                chat_id=user_id,
                message_id=status_msg.message_id
            )
            os.remove(temp_filename)
            return
        
        # Форматируем результаты
        result_text = format_for_telegram(results)
        
        # Создаем путь для сегментированного изображения
        segmented_path = segmented_img
        
        # Отправляем сегментированное изображение
        with open(segmented_path, 'rb') as f:
            bot.send_photo(user_id, f, caption="🔍 Сегментированное изображение")
        
        # Отправляем результаты текстом
        bot.send_message(user_id, result_text)
        
        # Удаляем сообщение о статусе
        bot.delete_message(user_id, status_msg.message_id)
        
        # Очищаем временные файлы
        os.remove(temp_filename)
        
        # Удаляем сегментированное изображение через 5 минут (не сейчас, чтобы успеть скачать)
        # В реальном проекте можно настроить очистку через cleanup_old_files
        
    except Exception as e:
        bot.edit_message_text(
            f"❌ Ошибка при анализе: {str(e)}",
            chat_id=user_id,
            message_id=status_msg.message_id
        )
        print(f"Ошибка для пользователя {user_id}: {e}")

# Обработка документов (на случай, если отправят как файл)
@bot.message_handler(content_types=['document'])
def handle_document(message):
    user_id = message.from_user.id
    
    # Проверяем, что это изображение
    if message.document.mime_type.startswith('image/'):
        # Обрабатываем как фото
        msg = bot.send_message(user_id, "🔄 Обрабатываю изображение...")
        
        try:
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            
            temp_filename = f"temp_images/{uuid.uuid4()}.jpg"
            with open(temp_filename, 'wb') as f:
                f.write(downloaded_file)
            
            # Здесь можно вызвать ту же логику, что и для фото
            # Для краткости пропустим, но в реальном проекте нужно добавить
            
            bot.edit_message_text(
                "✅ Функция обработки документов в разработке. Используйте отправку как фото.",
                chat_id=user_id,
                message_id=msg.message_id
            )
            
        except Exception as e:
            bot.edit_message_text(
                f"❌ Ошибка: {e}",
                chat_id=user_id,
                message_id=msg.message_id
            )
    else:
        bot.send_message(user_id, "❌ Пожалуйста, отправьте изображение в формате фото или JPEG/PNG")

# Обработка текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    user_id = message.from_user.id
    bot.send_message(
        user_id,
        "❓ Я понимаю только команды и фотографии.\n"
        "Используйте /start для начала работы."
    )

# ==================== ЗАПУСК БОТА ====================

if __name__ == "__main__":
    print("🤖 Бот запущен...")
    print(f"Токен: {BOT_TOKEN}")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        bot.infinity_polling()
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен")
    except Exception as e:
        print(f"❌ Ошибка: {e}")    
