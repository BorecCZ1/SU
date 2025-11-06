"""
Model training and prediction utilities
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pickle


class ModelTrainer:
    """Class for training and managing ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_linear_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train linear regression models (Linear, Ridge, Lasso, ElasticNet)
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary of trained models
        """
        linear_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=10.0),
            'Lasso': Lasso(alpha=0.001),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5)
        }
        
        for name, model in linear_models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"{name} trained successfully")
            
        return linear_models
    
    def train_tree_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train tree-based models (RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost)
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary of trained models
        """
        tree_models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=5,
                random_state=42,
                verbose=False
            )
        }
        
        for name, model in tree_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"{name} trained successfully")
            
        return tree_models
    
    def cross_validate(self, model, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X, y, 
            scoring='neg_root_mean_squared_error',
            cv=kfold,
            n_jobs=-1
        )
        return -scores  # Convert to positive RMSE
    
    def save_model(self, model, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


class EnsembleModel:
    """Class for ensemble methods"""
    
    def __init__(self, models: dict):
        """
        Initialize ensemble
        
        Args:
            models: Dictionary of trained models
        """
        self.models = models
        self.weights = None
        
    def simple_average(self, X):
        """Simple average of predictions"""
        predictions = np.column_stack([
            model.predict(X) for model in self.models.values()
        ])
        return predictions.mean(axis=1)
    
    def weighted_average(self, X, weights=None):
        """Weighted average of predictions"""
        if weights is None:
            weights = self.weights if self.weights is not None else [1/len(self.models)] * len(self.models)
            
        predictions = np.column_stack([
            model.predict(X) for model in self.models.values()
        ])
        return np.average(predictions, axis=1, weights=weights)


if __name__ == "__main__":
    print("Models module ready")

