import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import os
import json
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'models/model'
TEST_DATA_PATH = '/Users/artem/Documents/GitHub/ai_training/datasets/cian_data_cleaned.csv'
TARGET_COLUMN = 'sq_price_amt'
OUTPUT_DIR = 'evaluation_results'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_model(model_path: str) -> TabularPredictor:
    """Load the trained AutoGluon model"""
    print(f"Loading model from {model_path}...")
    try:
        predictor = TabularPredictor.load(model_path)
        print("Model loaded successfully!")
        return predictor
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first using train.py")


def load_test_data(data_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Load and split test data"""
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Check if target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Split data if we need to create a test set
    # In practice, you might want to use a separate test file
    print(f"Splitting data (test_size={test_size})...")
    _, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"Test set size: {len(test_data)}")
    
    return test_data


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Convert to percentage
    }
    
    # Additional metrics
    metrics['Mean_Error'] = np.mean(y_pred - y_true)
    metrics['Median_Absolute_Error'] = np.median(np.abs(y_pred - y_true))
    metrics['Max_Error'] = np.max(np.abs(y_pred - y_true))
    
    # Percentage of predictions within different error thresholds
    abs_errors = np.abs(y_pred - y_true)
    metrics['Within_10%'] = np.mean(abs_errors / np.abs(y_true) <= 0.10) * 100
    metrics['Within_20%'] = np.mean(abs_errors / np.abs(y_true) <= 0.20) * 100
    metrics['Within_30%'] = np.mean(abs_errors / np.abs(y_true) <= 0.30) * 100
    
    return metrics


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Create scatter plot of predictions vs actual values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: predictions_vs_actual.png")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Create residual plot"""
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: residuals_plot.png")


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Create error distribution plot"""
    abs_errors = np.abs(y_pred - y_true)
    relative_errors = (abs_errors / np.abs(y_true)) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute error distribution
    axes[0].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Absolute Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Relative error distribution
    axes[1].hist(relative_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Relative Error (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Relative Error Distribution (%)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: error_distribution.png")


def evaluate_model(
    model_path: str = MODEL_PATH,
    test_data_path: Optional[str] = None,
    test_data: Optional[pd.DataFrame] = None,
    target_column: str = TARGET_COLUMN,
    output_dir: str = OUTPUT_DIR,
    create_plots: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to test data CSV file (optional if test_data is provided)
        test_data: DataFrame with test data (optional if test_data_path is provided)
        target_column: Name of the target column
        output_dir: Directory to save evaluation results
        create_plots: Whether to create visualization plots
    
    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    predictor = load_model(model_path)
    
    # Load test data
    if test_data is None:
        if test_data_path is None:
            test_data_path = TEST_DATA_PATH
        test_data = load_test_data(test_data_path, target_column, TEST_SIZE, RANDOM_STATE)
    
    # Check if target column exists in test data
    if target_column not in test_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data")
    
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = predictor.predict(test_data.drop(columns=[target_column]))
    y_true = test_data[target_column].values
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    for metric_name, metric_value in metrics.items():
        if 'Within' in metric_name or 'MAPE' in metric_name:
            print(f"{metric_name:25s}: {metric_value:10.2f}%")
        else:
            print(f"{metric_name:25s}: {metric_value:15.4f}")
    
    # AutoGluon's built-in evaluation
    print("\n" + "="*60)
    print("AUTOGLUON EVALUATION")
    print("="*60)
    autogluon_performance = predictor.evaluate(test_data, silent=False)
    print(f"\nAutoGluon Performance:\n{autogluon_performance}")
    
    # Leaderboard
    print("\n" + "="*60)
    print("MODEL LEADERBOARD")
    print("="*60)
    leaderboard = predictor.leaderboard(test_data, silent=False)
    print(f"\n{leaderboard}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    try:
        feature_importance = predictor.feature_importance(test_data)
        print(f"\n{feature_importance.head(20)}")
    except Exception as e:
        print(f"Could not compute feature importance: {e}")
        feature_importance = None
    
    # Create visualizations
    if create_plots:
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        try:
            plot_predictions_vs_actual(y_true, y_pred, output_dir)
            plot_residuals(y_true, y_pred, output_dir)
            plot_error_distribution(y_true, y_pred, output_dir)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save metrics to JSON
    results = {
        'metrics': metrics,
        'autogluon_performance': autogluon_performance.to_dict() if hasattr(autogluon_performance, 'to_dict') else str(autogluon_performance),
        'test_set_size': len(test_data),
        'model_path': model_path
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {os.path.join(output_dir, 'evaluation_metrics.json')}")
    
    # Save metrics to text file
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Set Size: {len(test_data)}\n\n")
        
        f.write("METRICS:\n")
        f.write("-"*60 + "\n")
        for metric_name, metric_value in metrics.items():
            if 'Within' in metric_name or 'MAPE' in metric_name:
                f.write(f"{metric_name:25s}: {metric_value:10.2f}%\n")
            else:
                f.write(f"{metric_name:25s}: {metric_value:15.4f}\n")
        
        f.write("\n\nAUTOGLUON PERFORMANCE:\n")
        f.write("-"*60 + "\n")
        f.write(str(autogluon_performance))
        
        f.write("\n\nMODEL LEADERBOARD:\n")
        f.write("-"*60 + "\n")
        f.write(str(leaderboard))
        
        if feature_importance is not None:
            f.write("\n\nFEATURE IMPORTANCE (Top 20):\n")
            f.write("-"*60 + "\n")
            f.write(str(feature_importance.head(20)))
    
    print(f"Saved: {os.path.join(output_dir, 'evaluation_results.txt')}")
    
    # Save leaderboard to CSV
    if leaderboard is not None:
        leaderboard.to_csv(os.path.join(output_dir, 'leaderboard.csv'))
        print(f"Saved: {os.path.join(output_dir, 'leaderboard.csv')}")
    
    # Save feature importance to CSV
    if feature_importance is not None:
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'))
        print(f"Saved: {os.path.join(output_dir, 'feature_importance.csv')}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {os.path.abspath(output_dir)}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate AutoGluon model')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help=f'Path to trained model (default: {MODEL_PATH})')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data CSV file (default: uses train/test split)')
    parser.add_argument('--target', type=str, default=TARGET_COLUMN,
                        help=f'Target column name (default: {TARGET_COLUMN})')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory for results (default: {OUTPUT_DIR})')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip creating visualization plots')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        target_column=args.target,
        output_dir=args.output_dir,
        create_plots=not args.no_plots
    )

