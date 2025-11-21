
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import accuracy, root_mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoGluonTrainer:
    """Trainer class for AutoGluon models."""
    
    def __init__(
        self,
        train_data_path: str,
        label_column: str,
        problem_type: Optional[str] = None,
        output_dir: str = "models",
        time_limit: int = 3600,
        presets: str = "best_quality",
        eval_metric: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AutoGluon Trainer.
        
        Args:
            train_data_path: Path to training data (CSV, Parquet, etc.)
            label_column: Name of the target/label column
            problem_type: 'binary', 'multiclass', 'regression', or None (auto-detect)
            output_dir: Directory to save the trained model
            time_limit: Time limit for training in seconds (default: 1 hour)
            presets: AutoGluon presets ('best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment')
            eval_metric: Evaluation metric to use (auto-selected if None)
            **kwargs: Additional arguments to pass to TabularPredictor
        """
        self.train_data_path = train_data_path
        self.label_column = label_column
        self.problem_type = problem_type
        self.output_dir = output_dir
        self.time_limit = time_limit
        self.presets = presets
        self.eval_metric = eval_metric
        self.kwargs = kwargs
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.predictor = None
        self.train_data = None
        self.results = None
    
    def load_data(self) -> pd.DataFrame:
        """Load training data from file."""
        logger.info(f"Loading training data from {self.train_data_path}")
        
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
        
        # Support multiple file formats
        file_ext = Path(self.train_data_path).suffix.lower()
        
        if file_ext == '.csv':
            self.train_data = pd.read_csv(self.train_data_path)
        elif file_ext in ['.parquet', '.pq']:
            self.train_data = pd.read_parquet(self.train_data_path)
        elif file_ext in ['.xlsx', '.xls']:
            self.train_data = pd.read_excel(self.train_data_path)
        else:
            # Try CSV as default
            self.train_data = pd.read_csv(self.train_data_path)
        
        logger.info(f"Loaded {len(self.train_data)} rows and {len(self.train_data.columns)} columns")
        logger.info(f"Columns: {list(self.train_data.columns)}")
        
        # Validate label column exists
        if self.label_column not in self.train_data.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in data. "
                f"Available columns: {list(self.train_data.columns)}"
            )
        
        return self.train_data
    
    def detect_problem_type(self) -> str:
        """Auto-detect problem type from the label column."""
        if self.problem_type:
            return self.problem_type
        
        logger.info("Auto-detecting problem type...")
        
        label_data = self.train_data[self.label_column]
        n_unique = label_data.nunique()
        n_samples = len(label_data)
        
        # Check if numeric
        is_numeric = pd.api.types.is_numeric_dtype(label_data)
        
        if is_numeric:
            if n_unique == 2:
                problem_type = 'binary'
                logger.info("Detected binary classification")
            elif n_unique < 0.1 * n_samples and n_unique < 100:
                problem_type = 'multiclass'
                logger.info(f"Detected multiclass classification ({n_unique} classes)")
            else:
                problem_type = 'regression'
                logger.info("Detected regression")
        else:
            if n_unique == 2:
                problem_type = 'binary'
                logger.info("Detected binary classification")
            else:
                problem_type = 'multiclass'
                logger.info(f"Detected multiclass classification ({n_unique} classes)")
        
        return problem_type
    
    def get_eval_metric(self, problem_type: str) -> str:
        """Get appropriate evaluation metric for the problem type."""
        if self.eval_metric:
            return self.eval_metric
        
        metric_map = {
            'binary': 'accuracy',
            'multiclass': 'accuracy',
            'regression': 'root_mean_squared_error'
        }
        
        return metric_map.get(problem_type, 'accuracy')
    
    def train(self, holdout_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Train the AutoGluon model.
        
        Args:
            holdout_ratio: Ratio of data to use for validation (default: 0.2)
        
        Returns:
            Dictionary containing training results and metrics
        """
        # Load data
        self.load_data()
        
        # Detect problem type
        problem_type = self.detect_problem_type()
        eval_metric = self.get_eval_metric(problem_type)
        
        logger.info(f"Training AutoGluon model with:")
        logger.info(f"  - Problem type: {problem_type}")
        logger.info(f"  - Evaluation metric: {eval_metric}")
        logger.info(f"  - Time limit: {self.time_limit}s")
        logger.info(f"  - Presets: {self.presets}")
        logger.info(f"  - Label column: {self.label_column}")
        
        # Initialize predictor
        self.predictor = TabularPredictor(
            label=self.label_column,
            path=self.output_dir,
            problem_type=problem_type,
            eval_metric=eval_metric,
            **self.kwargs
        )
        
        # Train the model
        logger.info("Starting training...")
        self.results = self.predictor.fit(
            train_data=self.train_data,
            time_limit=self.time_limit,
            presets=self.presets
        )
        
        logger.info("Training completed!")
        logger.info(f"Best model: {self.results.get('best_model', 'N/A')}")
        
        # Get leaderboard
        leaderboard = self.predictor.leaderboard(silent=True)
        logger.info("\nModel Leaderboard:")
        logger.info(f"\n{leaderboard.to_string()}")
        
        # Evaluate on training data if holdout is used
        if holdout_ratio > 0:
            logger.info(f"\nEvaluating on {holdout_ratio*100}% holdout data...")
            test_data = self.train_data.sample(frac=holdout_ratio, random_state=42)
            train_data = self.train_data.drop(test_data.index)
            
            # Retrain with full data or evaluate separately
            y_pred = self.predictor.predict(test_data.drop(columns=[self.label_column]))
            y_true = test_data[self.label_column]
            
            if problem_type in ['binary', 'multiclass']:
                score = accuracy(y_true, y_pred)
                logger.info(f"Holdout Accuracy: {score:.4f}")
            else:
                score = root_mean_squared_error(y_true, y_pred)
                logger.info(f"Holdout RMSE: {score:.4f}")
        
        # Get feature importance
        try:
            feature_importance = self.predictor.feature_importance(self.train_data)
            logger.info("\nTop 10 Most Important Features:")
            logger.info(f"\n{feature_importance.head(10).to_string()}")
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
        
        return {
            'best_model': self.results.get('best_model'),
            'leaderboard': leaderboard.to_dict(),
            'problem_type': problem_type,
            'eval_metric': eval_metric,
            'model_path': self.output_dir
        }
    
    def save_model_info(self, output_path: Optional[str] = None):
        """Save model information to a file."""
        if self.results is None:
            logger.warning("No training results to save. Train the model first.")
            return
        
        output_path = output_path or os.path.join(self.output_dir, "model_info.txt")
        
        with open(output_path, 'w') as f:
            f.write("AutoGluon Model Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Problem Type: {self.problem_type}\n")
            f.write(f"Label Column: {self.label_column}\n")
            f.write(f"Best Model: {self.results.get('best_model', 'N/A')}\n")
            f.write(f"Training Time: {self.time_limit}s\n")
            f.write(f"Presets: {self.presets}\n")
            f.write(f"\nModel saved at: {self.output_dir}\n")
        
        logger.info(f"Model info saved to {output_path}")


def main():
    """Main training function with command-line interface."""
    parser = argparse.ArgumentParser(description='Train AutoGluon model')
    
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training data file (CSV, Parquet, etc.)'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        required=True,
        help='Name of the target/label column'
    )
    parser.add_argument(
        '--problem-type',
        type=str,
        choices=['binary', 'multiclass', 'regression'],
        default=None,
        help='Problem type (auto-detected if not specified)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save the trained model (default: models)'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=3600,
        help='Time limit for training in seconds (default: 3600)'
    )
    parser.add_argument(
        '--presets',
        type=str,
        default='best_quality',
        choices=['best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment'],
        help='AutoGluon presets (default: best_quality)'
    )
    parser.add_argument(
        '--eval-metric',
        type=str,
        default=None,
        help='Evaluation metric (auto-selected if not specified)'
    )
    parser.add_argument(
        '--holdout-ratio',
        type=float,
        default=0.2,
        help='Ratio of data to use for validation (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AutoGluonTrainer(
        train_data_path=args.train_data,
        label_column=args.label_column,
        problem_type=args.problem_type,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        presets=args.presets,
        eval_metric=args.eval_metric
    )
    
    # Train model
    results = trainer.train(holdout_ratio=args.holdout_ratio)
    
    # Save model info
    trainer.save_model_info()
    
    logger.info(f"\nTraining completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()

