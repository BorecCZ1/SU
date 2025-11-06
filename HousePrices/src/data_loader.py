"""
Data loading utilities
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Class for loading and basic data operations"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data
        
        Returns:
            Training dataframe with SalePrice
        """
        train_path = self.data_dir / "train.csv"
        return pd.read_csv(train_path)
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data
        
        Returns:
            Test dataframe without SalePrice
        """
        test_path = self.data_dir / "test.csv"
        return pd.read_csv(test_path)
    
    def load_both(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both train and test data
        
        Returns:
            Tuple of (train_df, test_df)
        """
        return self.load_train_data(), self.load_test_data()
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with basic statistics
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return info


if __name__ == "__main__":
    # Quick test
    loader = DataLoader()
    train, test = loader.load_both()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

