"""
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Class for data preprocessing operations"""
    
    def __init__(self):
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.label_encoders = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            strategy: Dictionary mapping column names to imputation strategies
            
        Returns:
            DataFrame with imputed values
        """
        # TODO: Implement missing value handling
        # Návrhy:
        # - Numerické: median/mean
        # - Kategorické: mode nebo 'None'
        # - Speciální případy: LotFrontage (based on Neighborhood), GarageYrBlt
        pass
    
    def encode_categorical(self, df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input dataframe
            method: Encoding method ('label', 'onehot', 'target')
            
        Returns:
            DataFrame with encoded categorical variables
        """
        # TODO: Implement categorical encoding
        pass
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input dataframe
            method: Scaling method ('standard', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        # TODO: Implement feature scaling
        pass
    
    def remove_outliers(self, df: pd.DataFrame, columns: list = None, method: str = 'zscore', 
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from dataset
        
        Args:
            df: Input dataframe
            columns: Columns to check for outliers (None = all numeric)
            method: Method for outlier detection ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame without outliers
        """
        # TODO: Implement outlier removal
        pass


if __name__ == "__main__":
    print("Preprocessing module ready")

