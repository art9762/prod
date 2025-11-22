import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# --- НАСТРОЙКИ ---
# Имя вашего файла с данными для обучения
TRAIN_CSV_PATH = '/Users/artem/Documents/GitHub/ai_training/datasets/cian_data_cleaned.csv'

# Имя колонки, которую нужно предсказывать (например, 'price', 'is_churn', 'species')
TARGET_COLUMN = 'sq_price_amt' 

# Папка, куда сохранится обученная модель
MODEL_SAVE_PATH = 'models/model'

# Качество обучения. Опции: 'best_quality' (долго, но точно), 'high_quality', 'good_quality', 'medium_quality'
PRESETS = 'best_quality' 
# -----------------

def train_model():
    print("Загрузка данных...")
    train_data = TabularDataset(TRAIN_CSV_PATH)
    
    print(f"Начинаем обучение. Целевая колонка: {TARGET_COLUMN}")
    
    # Инициализация предиктора
    predictor = TabularPredictor(
        label=TARGET_COLUMN, 
        path=MODEL_SAVE_PATH,
        problem_type=None,  # AutoGluon сам определит (регрессия или классификация)
        eval_metric=None    # AutoGluon сам выберет метрику (accuracy, rmse и т.д.)
    )
    
    # Запуск обучения
    predictor.fit(
        train_data, 
        presets=PRESETS
    )

    print("Обучение завершено.")
    
    # Вывод результатов лидерборда (сравнение моделей)
    print("\nРезультаты моделей:")
    leaderboard = predictor.leaderboard(train_data, silent=True)
    print(leaderboard.head())

if __name__ == '__main__':
    train_model()