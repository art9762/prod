import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# --- НАСТРОЙКИ ---
# Имя файла с новыми данными (без целевой колонки или с ней для проверки)
TEST_CSV_PATH = 'test_data.csv'

# Папка, где лежит модель (должна совпадать с MODEL_SAVE_PATH из train.py)
MODEL_PATH = 'autogluon_model'

# Имя файла, куда сохранятся результаты
OUTPUT_FILENAME = 'predictions.csv'
# -----------------

# FastAPI app
app = FastAPI(title="Prediction API", description="API for making predictions with AutoGluon model")

# Global predictor variable
predictor: Optional[TabularPredictor] = None


def load_model():
    """Load the model at startup"""
    global predictor
    print(f"Загрузка модели из {MODEL_PATH}...")
    try:
        predictor = TabularPredictor.load(MODEL_PATH)
        print("Модель успешно загружена!")
        return predictor
    except FileNotFoundError:
        print(f"Ошибка: Модель не найдена в {MODEL_PATH}. Сначала запустите train.py!")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model when the API starts"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Prediction API is running", "model_path": MODEL_PATH}


@app.get("/health")
async def health():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """
    Make predictions on a single record or batch of records.
    
    Accepts JSON in the following formats:
    - Single record: {"feature1": value1, "feature2": value2, ...}
    - Batch: {"records": [{"feature1": value1, ...}, {"feature1": value2, ...}]}
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the model exists.")
    
    try:
        # Check if it's a batch request
        if "records" in data:
            records = data["records"]
            if not isinstance(records, list) or len(records) == 0:
                raise HTTPException(status_code=400, detail="'records' must be a non-empty list")
            df = pd.DataFrame(records)
        else:
            # Single record
            df = pd.DataFrame([data])
        
        # Make predictions
        predictions = predictor.predict(df)
        
        # Convert predictions to list (handle both single and multiple predictions)
        if len(predictions) == 1:
            result = {"prediction": float(predictions.iloc[0]) if hasattr(predictions.iloc[0], '__float__') else predictions.iloc[0]}
        else:
            result = {"predictions": predictions.tolist()}
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(data: List[Dict[str, Any]]):
    """
    Make predictions on a batch of records.
    
    Accepts JSON array of records:
    [{"feature1": value1, "feature2": value2, ...}, ...]
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the model exists.")
    
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=400, detail="Request body must be a non-empty list of records")
    
    try:
        df = pd.DataFrame(data)
        predictions = predictor.predict(df)
        
        result = {
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making predictions: {str(e)}")


def make_predictions():
    """Original function for command-line predictions from CSV file"""
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
    import sys
    
    # Check if running as API server or CLI tool
    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        # Run as API server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run original CLI function
        make_predictions()