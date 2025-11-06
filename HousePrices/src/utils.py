"""
Utility functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of dataframe by downcasting numeric types
    
    Args:
        df: Input dataframe
        verbose: Print memory reduction info
        
    Returns:
        DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def save_submission(ids, predictions, filename: str = 'submission.csv'):
    """
    Create submission file for Kaggle
    
    Args:
        ids: Test IDs
        predictions: Predicted values
        filename: Output filename
    """
    submission = pd.DataFrame({
        'Id': ids,
        'SalePrice': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def plot_correlation_heatmap(df: pd.DataFrame, figsize=(15, 12), top_n: int = None):
    """
    Plot correlation heatmap
    
    Args:
        df: Input dataframe
        figsize: Figure size
        top_n: Show only top N correlated features with target (if exists)
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if top_n and 'SalePrice' in numeric_df.columns:
        # Get top N correlated features with SalePrice
        correlations = numeric_df.corr()['SalePrice'].abs().sort_values(ascending=False)
        top_features = correlations.head(top_n).index
        numeric_df = numeric_df[top_features]
    
    plt.figure(figsize=figsize)
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_missing_values(df: pd.DataFrame, threshold: int = 0):
    """
    Visualize missing values
    
    Args:
        df: Input dataframe
        threshold: Only show columns with more than threshold missing values
    """
    missing = df.isnull().sum()
    missing = missing[missing > threshold].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values!")
        return
    
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.title('Missing Values by Column', fontsize=14)
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Number of Missing Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")
    print(f"Percentage: {100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2f}%")


if __name__ == "__main__":
    print("Utils module ready")

