"""
Model evaluation utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """Class for model evaluation and visualization"""
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name: str = None) -> dict:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model (optional)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        if model_name:
            self.results[model_name] = metrics
            
        return metrics
    
    def print_metrics(self, metrics: dict, model_name: str = None):
        """Print metrics in a nice format"""
        if model_name:
            print(f"\n{'='*50}")
            print(f"Metrics for {model_name}")
            print(f"{'='*50}")
        
        for metric_name, value in metrics.items():
            print(f"{metric_name:10s}: {value:.4f}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models
        
        Returns:
            DataFrame with comparison
        """
        if not self.results:
            print("No models evaluated yet")
            return None
            
        df = pd.DataFrame(self.results).T
        df = df.sort_values('RMSE')
        return df
    
    def plot_predictions_vs_actual(self, y_true, y_pred, title: str = "Predictions vs Actual"):
        """
        Plot predicted vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title: str = "Residual Plot"):
        """
        Plot residuals
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residuals vs Predicted', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black')
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residuals Distribution', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n: int = 20):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            top_n: Number of top features to show
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model doesn't have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Evaluation module ready")

