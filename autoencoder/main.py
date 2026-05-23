"""
Недвижимость Москвы: Классификатор квартир
Инструмент для определения типа квартиры, её кластера и аномальности
"""

import tkinter as tk
from tkinter import ttk, messagebox
import umap
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ============================================================
# ЗАГРУЗКА МОДЕЛЕЙ (при запуске приложения)
# ============================================================

import joblib
import json
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

MODELS_PATH = "saved_models/"

print("Загрузка моделей...")

# ============================================================
# 1. Воссоздаём архитектуру вручную (укажите ваши параметры!)
# ============================================================
INPUT_DIM = 21      # ← ЗАМЕНИТЕ на ваше значение (из Colab)
ENCODING_DIM = 8    # ← первый скрытый слой
BOTTLENECK_DIM = 4  # ← бутылочное горлышко

# Автоэнкодер
input_layer = Input(shape=(INPUT_DIM,))
encoded = Dense(ENCODING_DIM, activation='relu', name='dense')(input_layer)
encoded = Dense(BOTTLENECK_DIM, activation='relu', name='bottleneck')(encoded)
decoded = Dense(ENCODING_DIM, activation='relu', name='dense_2')(encoded)
decoded = Dense(INPUT_DIM, activation='linear', name='output')(decoded)
autoencoder = Model(input_layer, decoded, name='autoencoder')

# Энкодер (до bottleneck)
encoder_model = Model(inputs=input_layer, outputs=encoded, name='encoder')

# Компилируем
autoencoder.compile(optimizer='adam', loss='mse')

# ============================================================
# 2. Загружаем веса
# ============================================================
autoencoder.load_weights(MODELS_PATH + 'autoencoder.weights.h5')
encoder_model.load_weights(MODELS_PATH + 'encoder.weights.h5')

print("✅ Модели автоэнкодера загружены!")

# ============================================================
# 3. Загружаем остальные компоненты
# ============================================================
preprocessor = joblib.load(MODELS_PATH + "preprocessor.pkl")
umap_model = joblib.load(MODELS_PATH + "umap_model.pkl")
cluster_labels = np.load(MODELS_PATH + "cluster_labels.npy")
umap_coords_train = np.load(MODELS_PATH + "umap_coords.npy")

with open(MODELS_PATH + "threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

print("✅ Все модели загружены!")

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

# Словарь с описанием кластеров (на основе вашего анализа)
CLUSTER_DESCRIPTIONS = {
    -1: {
        "name": "Аномальные / уникальные объекты",
        "description": "Очень старые дома (до 1950 г.), часто в центре, но с плохой транспортной доступностью. Смесь новостроек и апартаментов. Требуют ручной проверки.",
        "color": "#FF6B6B",
        "recommendation": "⚠️ Уникальный объект — рекомендуем личную проверку"
    },
    0: {
        "name": "Массовый рынок Москвы",
        "description": "Сбалансированные квартиры со средней площадью ~88 м². Смесь вторички и новостроек. Представлены все округа.",
        "color": "#4ECDC4",
        "recommendation": "✅ Обычная квартира, можно рассматривать"
    },
    1: {
        "name": "Элитные высотные комплексы",
        "description": "Небоскрёбы (средняя этажность 54 этажа!), большие площади (~107 м²). Расположены в ЦАО и СЗАО. Высокая вариативность цен.",
        "color": "#F7B731",
        "recommendation": "🏙️ Премиум-сегмент — внимательно проверьте цену и вид из окна"
    },
    2: {
        "name": "Малогабаритная вторичка",
        "description": "Небольшие квартиры (~54 м²) в старых домах (1960-е). 100% вторичка. Самый предсказуемый сегмент.",
        "color": "#9B59B6",
        "recommendation": "🏠 Бюджетный вариант, подходит для старта или сдачи в аренду"
    }
}


def get_cluster_prediction(new_umap_point):
    """
    Определяет кластер для новой точки на основе ближайшего соседа из обучающей выборки.
    (DBSCAN не имеет метода predict, поэтому используем NearestNeighbors)
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(umap_coords_train)
    distances, indices = nn.kneighbors(new_umap_point)
    return cluster_labels[indices[0][0]]


def predict_anomaly(reconstruction_error):
    """Определяет, является ли объект аномальным"""
    return reconstruction_error > threshold


def make_prediction(data_dict):
    """
    data_dict: словарь с ключами-названиями признаков и значениями
    Возвращает: cluster, is_anomaly, error, description
    """
    # 1. Преобразуем в DataFrame с правильными колонками
    input_df = pd.DataFrame([data_dict])

    # 2. Применяем предобработку (трансформируем)
    try:
        processed = preprocessor.transform(input_df)
    except Exception as e:
        raise ValueError(f"Ошибка в предобработке: {e}. Проверьте введённые данные.")

    # 3. Получаем латентное пространство через энкодер
    latent = encoder_model.predict(processed, verbose=0)

    # 4. Получаем UMAP-координаты
    umap_point = umap_model.transform(latent)

    # 5. Определяем кластер через ближайшего соседа
    cluster = get_cluster_prediction(umap_point)

    # 6. Считаем ошибку реконструкции (аномалию)
    reconstructed = autoencoder.predict(processed, verbose=0)
    error = float(np.mean(np.square(processed - reconstructed)))
    is_anomaly = predict_anomaly(error)

    # 7. Получаем описание кластера
    cluster_info = CLUSTER_DESCRIPTIONS.get(cluster, {
        "name": "Неизвестный кластер",
        "description": "Нет описания",
        "color": "#CCCCCC",
        "recommendation": "Обратитесь к администратору"
    })

    return {
        'cluster': cluster,
        'cluster_name': cluster_info['name'],
        'cluster_description': cluster_info['description'],
        'cluster_recommendation': cluster_info['recommendation'],
        'is_anomaly': is_anomaly,
        'error': round(error, 4),
        'threshold': round(threshold, 4),
        'color': cluster_info['color']
    }


# ============================================================
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ============================================================

class RealEstateClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор квартир Москвы")
        self.root.geometry("900x750")
        self.root.resizable(False, False)

        # Устанавливаем стиль
        style = ttk.Style()
        style.theme_use('clam')

        # Основные рамки
        self.create_input_frame()
        self.create_result_frame()
        self.create_button_frame()

    def create_input_frame(self):
        """Создаёт форму ввода данных"""
        input_frame = ttk.LabelFrame(self.root, text="Характеристики квартиры", padding=10)
        input_frame.pack(fill="both", padx=10, pady=10)

        self.fields = {}
        fields_config = [
            ("Цена (руб)", "price", "entry"),  # ← ДОБАВЛЕНО
            ("Количество комнат", "number_of_rooms", "entry"),
            ("Общая площадь (м²)", "total_area", "entry"),
            ("Жилая площадь (м²)", "living_area", "entry"),
            ("Этаж", "floor", "entry"),
            ("Всего этажей в доме", "number_of_floors", "entry"),
            ("Год постройки", "construction_year", "entry"),
            ("Время до метро (мин)", "min_to_metro", "entry"),
            ("Округ Москвы", "region_of_moscow", "combobox"),
            ("Тип жилья", "is_new", "combobox"),
            ("Является апартаментом", "is_apartments", "combobox")
        ]

        for i, (label, key, widget_type) in enumerate(fields_config):
            ttk.Label(input_frame, text=label + ":").grid(row=i, column=0, sticky="w", padx=5, pady=5)

            if widget_type == "entry":
                entry = ttk.Entry(input_frame, width=30)
                entry.grid(row=i, column=1, padx=5, pady=5)
                self.fields[key] = entry
            elif widget_type == "combobox":
                if key == "region_of_moscow":
                    values = ["ВАО", "ЗАО", "САО", "СВАО", "СЗАО", "ЦАО", "ЮАО", "ЮВАО", "ЮЗАО"]
                elif key == "is_new":
                    values = ["0 (вторичка)", "1 (новостройка)"]
                elif key == "is_apartments":
                    values = ["0 (квартира)", "1 (апартаменты)"]
                else:
                    values = []

                combo = ttk.Combobox(input_frame, values=values, width=27, state="readonly")
                combo.grid(row=i, column=1, padx=5, pady=5)
                self.fields[key] = combo

        ttk.Label(input_frame, text="⚠️ Заполните ВСЕ поля", foreground="orange").grid(row=len(fields_config), column=0,
                                                                                       columnspan=2, pady=10)

    def create_result_frame(self):
        """Создаёт область для вывода результатов"""
        result_frame = ttk.LabelFrame(self.root, text="Результат анализа", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.result_text = tk.Text(result_frame, height=15, width=80, font=("Arial", 11), wrap="word")
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Скроллбар
        scrollbar = ttk.Scrollbar(self.result_text, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    def create_button_frame(self):
        """Создаёт кнопки управления"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(button_frame, text="📊 Анализировать квартиру", command=self.on_analyze, width=25).pack(side="left",
                                                                                                          padx=5)
        ttk.Button(button_frame, text="🗑️ Очистить форму", command=self.on_clear, width=20).pack(side="left", padx=5)
        ttk.Button(button_frame, text="❌ Выход", command=self.root.quit, width=15).pack(side="right", padx=5)

    def get_input_data(self):
        """Собирает данные из формы в словарь"""
        try:
            data = {}

            # Числовые поля
            data['price'] = float(self.fields['price'].get())  # ← ДОБАВЛЕНО
            data['number_of_rooms'] = float(self.fields['number_of_rooms'].get())
            data['total_area'] = float(self.fields['total_area'].get())
            data['living_area'] = float(self.fields['living_area'].get())
            data['floor'] = float(self.fields['floor'].get())
            data['number_of_floors'] = float(self.fields['number_of_floors'].get())
            data['construction_year'] = float(self.fields['construction_year'].get())
            data['min_to_metro'] = float(self.fields['min_to_metro'].get())

            # Категориальные поля
            region = self.fields['region_of_moscow'].get()
            if not region:
                raise ValueError("Выберите округ")
            data['region_of_moscow'] = region

            new_val = self.fields['is_new'].get()
            if not new_val:
                raise ValueError("Выберите тип жилья")
            data['is_new'] = float(new_val.split()[0])  # "0 (вторичка)" -> 0.0

            apt_val = self.fields['is_apartments'].get()
            if not apt_val:
                raise ValueError("Выберите тип недвижимости")
            data['is_apartments'] = float(apt_val.split()[0])  # "0 (квартира)" -> 0.0

            return data

        except ValueError as e:
            if "could not convert string to float" in str(e):
                raise ValueError("Пожалуйста, заполните все числовые поля целыми или дробными числами")
            raise e
        except Exception as e:
            raise ValueError(f"Ошибка в данных: {e}")

    def on_analyze(self):
        """Обработчик нажатия на кнопку анализа"""
        try:
            # Получаем данные
            data = self.get_input_data()

            # Делаем предсказание
            result = make_prediction(data)

            # Форматируем вывод
            self.result_text.configure(state="normal")
            self.result_text.delete(1.0, tk.END)

            # Заголовок
            self.result_text.insert(tk.END, "=" * 60 + "\n", "header")
            self.result_text.insert(tk.END, "РЕЗУЛЬТАТ КЛАССИФИКАЦИИ КВАРТИРЫ\n", "header")
            self.result_text.insert(tk.END, "=" * 60 + "\n\n", "header")

            # Основная информация
            self.result_text.insert(tk.END, f"📌 Кластер: {result['cluster']}\n", "title")
            self.result_text.insert(tk.END, f"   {result['cluster_name']}\n\n", "text")

            self.result_text.insert(tk.END, f"📝 Описание: {result['cluster_description']}\n\n", "text")

            # Аномалия
            if result['is_anomaly']:
                self.result_text.insert(tk.END, "⚠️ СТАТУС: АНОМАЛЬНЫЙ ОБЪЕКТ ⚠️\n", "anomaly")
                self.result_text.insert(tk.END,
                                        f"   Ошибка реконструкции: {result['error']} (порог: {result['threshold']})\n",
                                        "text")
                self.result_text.insert(tk.END,
                                        "   Значение выше порога → квартира имеет нестандартные характеристики\n\n",
                                        "text")
            else:
                self.result_text.insert(tk.END, "✅ СТАТУС: НОРМАЛЬНЫЙ ОБЪЕКТ\n", "normal")
                self.result_text.insert(tk.END,
                                        f"   Ошибка реконструкции: {result['error']} (порог: {result['threshold']})\n\n",
                                        "text")

            # Рекомендация
            self.result_text.insert(tk.END, "💡 Рекомендация:\n", "title")
            self.result_text.insert(tk.END, f"   {result['cluster_recommendation']}\n\n", "text")

            # Финал
            self.result_text.insert(tk.END, "=" * 60 + "\n", "footer")
            self.result_text.insert(tk.END, "Данные проанализированы нейросетью (автоэнкодер + UMAP + DBSCAN)",
                                    "footer")

            # Настраиваем цветовую схему текста
            self.result_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#2C3E50")
            self.result_text.tag_configure("title", font=("Arial", 11, "bold"))
            self.result_text.tag_configure("text", font=("Arial", 10))
            self.result_text.tag_configure("anomaly", font=("Arial", 11, "bold"), foreground="#E74C3C")
            self.result_text.tag_configure("normal", font=("Arial", 11, "bold"), foreground="#27AE60")
            self.result_text.tag_configure("footer", font=("Arial", 9, "italic"), foreground="#7F8C8D")

            self.result_text.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_clear(self):
        """Очищает форму ввода"""
        for field in self.fields.values():
            if isinstance(field, ttk.Entry):
                field.delete(0, tk.END)
            elif isinstance(field, ttk.Combobox):
                field.set('')


# ============================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = RealEstateClassifierApp(root)
    root.mainloop()