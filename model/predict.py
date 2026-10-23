"""Prediction API and CSV CLI; importing this module never loads a saved model."""

import argparse
from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", "models/model")
TEST_CSV_PATH = "test_data.csv"
OUTPUT_FILENAME = "predictions.csv"
MAX_BATCH_RECORDS = 1000
logger = logging.getLogger(__name__)


def resolve_model_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model(path=MODEL_PATH):
    # AutoGluon is needed for inference, not for API discovery or --help.
    from autogluon.tabular import TabularPredictor

    return TabularPredictor.load(str(resolve_model_path(path)))


def _records(value):
    if not isinstance(value, list) or not value or len(value) > MAX_BATCH_RECORDS:
        raise HTTPException(400, f"Provide between 1 and {MAX_BATCH_RECORDS} records")
    if any(not isinstance(row, dict) or not row for row in value):
        raise HTTPException(400, "Each record must be a non-empty feature object")
    return value


def _numbers(predictor, records):
    values = predictor.predict(pd.DataFrame.from_records(records))
    try:
        values = values.tolist() if hasattr(values, "tolist") else list(values)
        if len(values) != len(records):
            raise RuntimeError("Prediction count differs from input count")
        result = []
        for value in values:
            if pd.api.types.is_bool(value):
                raise RuntimeError("The regression model returned a boolean")
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("The model returned a non-finite prediction")
            result.append(number)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("The regression model returned invalid numeric output") from None
    return result


def create_app(*, model_path=MODEL_PATH, predictor_loader=None):
    loader = predictor_loader or load_model

    @asynccontextmanager
    async def lifespan(application):
        try:
            application.state.predictor = loader(model_path)
        except Exception as error:
            application.state.predictor = None
            logger.error("Model unavailable at startup (%s)", type(error).__name__)
        yield
        application.state.predictor = None

    application = FastAPI(title="Prediction API", lifespan=lifespan)
    application.state.predictor = None
    application.state.prediction_lock = threading.Lock()

    def run(records):
        predictor = application.state.predictor
        if predictor is None:
            raise HTTPException(503, "Model not loaded. Check MODEL_PATH and train the model first.")
        records = _records(records)
        try:
            # Sync routes run in FastAPI's worker pool; one model is not invoked
            # concurrently and expensive inference does not block the event loop.
            with application.state.prediction_lock:
                return _numbers(predictor, records)
        except (KeyError, ValueError, TypeError):
            raise HTTPException(400, "Features do not match the loaded regression model") from None
        except Exception as error:
            logger.error("Prediction failed (%s)", type(error).__name__)
            raise HTTPException(500, "The model could not produce valid predictions") from None

    @application.get("/")
    def root():
        return {"message": "Prediction API is running", "model_path": str(model_path)}

    @application.get("/health")
    def health():
        if application.state.predictor is None:
            raise HTTPException(503, "Model not loaded")
        return {"status": "healthy", "model_loaded": True}

    @application.post("/predict")
    def predict(data: dict[str, Any]):
        if "records" in data:
            values = run(data["records"])
            return {"predictions": values, "count": len(values)}
        return {"prediction": run([data])[0]}

    @application.post("/predict/batch")
    def predict_batch(data: list[dict[str, Any]]):
        values = run(data)
        return {"predictions": values, "count": len(values)}

    return application


app = create_app()


def make_predictions(input_path=TEST_CSV_PATH, output_path=OUTPUT_FILENAME,
                     model_path=MODEL_PATH, *, predictor_loader=None):
    destination = Path(output_path)
    if destination.resolve() == Path(input_path).resolve() or (
        destination.exists() and os.path.samefile(input_path, destination)
    ):
        raise ValueError("Output must be different from the input CSV")
    predictor = (predictor_loader or load_model)(model_path)
    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError("Input CSV contains no records")
    values = _numbers(predictor, frame.to_dict(orient="records"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=destination.parent, prefix=".prediction-",
                                         suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            pd.DataFrame({"prediction": values}).to_csv(file, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["predict", "serve"], default="predict")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=TEST_CSV_PATH)
    parser.add_argument("--output", default=OUTPUT_FILENAME)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run(create_app(model_path=args.model_path), host=args.host, port=args.port)
        return 0
    try:
        output = make_predictions(args.input, args.output, args.model_path)
    except Exception as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(f"Predictions saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
