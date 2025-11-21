import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# --- НАСТРОЙКИ ---
# Имя файла с новыми данными (без целевой колонки или с ней для проверки)
TEST_CSV_PATH = 'test_data.csv'

# Папка, где лежит модель (должна совпадать с MODEL_SAVE_PATH из train.py)
MODEL_PATH = 'autogluon_model'

# Имя файла, куда сохранятся результаты
OUTPUT_FILENAME = 'predictions.csv'
# -----------------

def make_predictions():
    print(f"Загрузка модели из {MODEL_PATH}...")
    try:
        predictor = TabularPredictor.load(MODEL_PATH)
    except FileNotFoundError:
        print("Ошибка: Модель не найдена. Сначала запустите train.py!")
        return

    print(f"Загрузка данных для предсказания из {TEST_CSV_PATH}...")
    test_data = TabularDataset(TEST_CSV_PATH)

    # Если в тестовых данных есть колонка с ответами, можно оценить точность
    # if predictor.label in test_data.columns:
    #     print("Оценка точность на тестовых данных:")
    #     performance = predictor.evaluate(test_data)
    #     print(performance)

    print("Генерация предсказаний...")
    # Получаем предсказания
    predictions = predictor.predict(test_data)
    
    # Если нужна вероятность классов (для классификации), раскомментируйте строку ниже:
    # prediction_proba = predictor.predict_proba(test_data)

    # Сохранение результатов
    # Мы объединяем исходные данные (или только ID) с предсказаниями для удобства
    output_df = pd.DataFrame({'prediction': predictions})
    
    # Если хотите сохранить вместе с исходными данными:
    # output_df = pd.concat([test_data, output_df], axis=1)

    output_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Предсказания сохранены в файл: {OUTPUT_FILENAME}")

if __name__ == '__main__':
    make_predictions()