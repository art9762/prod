from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

training_path = '/Users/artem/Documents/GitHub/ai_training/datasets/realty_picks.csv'
model_path = 'models/model'
target_column = 'monthly_price_rub'
presets = 'best_quality'

# Load data
print("Loading data...")
data = pd.read_csv(training_path)

# Split data into train and test sets
print("Splitting data into train and test sets...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")

# Train model
print("Training model...")
predictor = TabularPredictor(label=target_column, path=model_path).fit(
    train_data=train_data,
    time_limit=300,  
    presets='best_quality'  #'medium_quality', 'high_quality', 'best_quality'
)

# Evaluate model on test set
print("\n" + "="*50)
print("Evaluating model on test set...")
print("="*50)
performance = predictor.evaluate(test_data, silent=False)
print(f"\nEvaluation Results:")
print(performance)

# Get leaderboard to see individual model performance
print("\n" + "="*50)
print("Model Leaderboard:")
print("="*50)
leaderboard = predictor.leaderboard(test_data, silent=False)
print(leaderboard)

# Save evaluation results
eval_results_path = os.path.join(model_path, 'evaluation_results.txt')
os.makedirs(model_path, exist_ok=True)
with open(eval_results_path, 'w') as f:
    f.write("Evaluation Results:\n")
    f.write(str(performance))
    f.write("\n\nLeaderboard:\n")
    f.write(str(leaderboard))

print(f"\nEvaluation results saved to: {eval_results_path}")
print("\nTraining and evaluation completed!")

