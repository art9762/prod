# AI Training - Real Estate Price Prediction

A machine learning project for predicting real estate prices using AutoGluon's tabular prediction capabilities. This project includes model training, evaluation, and a REST API for making predictions.

## Overview

This project uses AutoGluon to train and deploy a regression model that predicts real estate prices (`sq_price_amt`) based on various property features. The model is trained on cleaned real estate data and can be used via a FastAPI-based REST API or command-line interface.

## Features

- **Automated ML Training**: Uses AutoGluon's TabularPredictor for automated model selection and training
- **Comprehensive Evaluation**: Detailed metrics and visualizations for model performance
- **REST API**: FastAPI-based API for real-time predictions
- **Batch Predictions**: Support for single and batch prediction requests
- **Data Cleaning**: Scripts for preprocessing and cleaning real estate datasets

## Project Structure

```
ai_training/
├── model/
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction API and CLI tool
│   └── evaluate.py       # Model evaluation script
├── datasets/
│   ├── cian_data.csv     # Raw real estate data
│   ├── cian_data_cleaned.csv  # Cleaned dataset
│   ├── dataset_clr.py    # Data cleaning script
│   └── ...
├── guides/               # Documentation and guides
├── requirements.txt      # Python dependencies
├── DOCKERFILE           # Docker configuration
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai_training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Before training, ensure your data is cleaned and prepared. The project includes a data cleaning script:

```bash
python datasets/dataset_clr.py
```

This script filters out outliers and prepares the dataset for training.

### 2. Model Training

Train the model using the training script:

```bash
python model/train.py
```

**Configuration** (in `train.py`):
- `training_path`: Path to cleaned training data
- `model_path`: Directory where the model will be saved
- `target_column`: Target variable to predict (default: `sq_price_amt`)
- `presets`: Quality preset (`'medium_quality'`, `'high_quality'`, `'best_quality'`)
- `time_limit`: Training time limit in seconds (default: 300)

The script will:
- Load and split the data (80% train, 20% test)
- Train multiple models using AutoGluon
- Evaluate on the test set
- Save the model and evaluation results

### 3. Model Evaluation

Evaluate the trained model with comprehensive metrics and visualizations:

```bash
python model/evaluate.py
```

**Options**:
```bash
python model/evaluate.py \
    --model_path models/model \
    --test_data datasets/cian_data_cleaned.csv \
    --target sq_price_amt \
    --output_dir evaluation_results \
    --no-plots  # Skip visualization plots
```

The evaluation script generates:
- **Metrics**: RMSE, MAE, R², MAPE, and more
- **Visualizations**: 
  - Predictions vs Actual scatter plot
  - Residual plots
  - Error distribution plots
- **Reports**: JSON and text files with detailed results
- **Leaderboard**: Performance comparison of all trained models
- **Feature Importance**: Top features contributing to predictions

### 4. Making Predictions

#### Option A: REST API (Recommended)

Start the prediction API server:

```bash
python model/predict.py serve
```

The API will be available at `http://localhost:8000`

**API Endpoints**:

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single or batch predictions

**Example: Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "feature1": value1,
       "feature2": value2,
       ...
     }'
```

**Example: Batch Prediction**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {"feature1": value1, "feature2": value2, ...},
       {"feature1": value3, "feature2": value4, ...}
     ]'
```

**Response Format**:
```json
{
  "prediction": 12345.67
}
```

or for batch:
```json
{
  "predictions": [12345.67, 23456.78, ...],
  "count": 2
}
```

#### Option B: Command Line

Make predictions from a CSV file:

```bash
python model/predict.py
```

**Configuration** (in `predict.py`):
- `TEST_CSV_PATH`: Path to CSV file with features (without target column)
- `MODEL_PATH`: Path to trained model
- `OUTPUT_FILENAME`: Output file for predictions

## Configuration

### Model Training Settings

Edit `model/train.py` to customize:
- Training data path
- Model save location
- Target column name
- Quality presets
- Training time limit

### API Settings

Edit `model/predict.py` to customize:
- Model path
- Test data path
- Output filename
- API host and port (in `uvicorn.run()`)

### Evaluation Settings

Edit `model/evaluate.py` to customize:
- Model path
- Test data path
- Target column
- Output directory
- Plot generation

## Dependencies

- **fastapi**: Web framework for the prediction API
- **uvicorn**: ASGI server for FastAPI
- **tensorflow**: Deep learning framework (used by AutoGluon)
- **autogluon**: Automated machine learning library
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning utilities

## Model Performance

The model performance depends on:
- Data quality and size
- Feature engineering
- Training time limit
- Quality preset selected

Check the evaluation results in `evaluation_results/` directory after running `evaluate.py` for detailed metrics.

## Docker Support

A Dockerfile is included for containerized deployment. Build and run:

```bash
docker build -t ai-training .
docker run -p 8000:8000 ai-training
```

## Notes

- The model uses AutoGluon's ensemble of multiple algorithms
- Training time increases with `best_quality` preset
- Ensure sufficient memory for large datasets
- The API loads the model at startup, so first request may be slower

## Troubleshooting

**Model not found error**:
- Ensure you've trained the model first using `train.py`
- Check that `MODEL_PATH` in `predict.py` matches the path used in training

**Memory issues**:
- Reduce dataset size or use `medium_quality` preset
- Increase system RAM or use a machine with more memory

**API connection errors**:
- Check that the API server is running
- Verify the port (default: 8000) is not in use

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]

