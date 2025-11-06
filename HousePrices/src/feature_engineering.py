"""
Feature engineering utilities
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Class for creating new features"""
    
    def __init__(self):
        self.created_features = []
        
    def create_total_sf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create total square footage feature
        
        TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
        """
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        self.created_features.append('TotalSF')
        return df
    
    def create_total_bathrooms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create total bathrooms feature
        
        TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
        """
        df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                                df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
        self.created_features.append('TotalBathrooms')
        return df
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features
        
        HouseAge = YrSold - YearBuilt
        RemodAge = YrSold - YearRemodAdd
        """
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        
        self.created_features.extend(['HouseAge', 'RemodAge', 'IsRemodeled'])
        return df
    
    def create_total_porch_sf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create total porch square footage
        
        TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
        """
        df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + 
                             df['3SsnPorch'] + df['ScreenPorch'])
        self.created_features.append('TotalPorchSF')
        return df
    
    def create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary features for presence of certain amenities
        """
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
        df['HasWoodDeck'] = (df['WoodDeckSF'] > 0).astype(int)
        
        binary_features = ['HasPool', 'HasGarage', 'HasBasement', 
                          'HasFireplace', 'Has2ndFloor', 'HasWoodDeck']
        self.created_features.extend(binary_features)
        return df
    
    def create_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite quality features
        """
        df['QualityIndex'] = df['OverallQual'] * df['OverallCond']
        df['GarageScore'] = df['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
        df['KitchenScore'] = df['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
        
        self.created_features.extend(['QualityIndex', 'GarageScore', 'KitchenScore'])
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        """
        # Area * Quality interactions
        df['GrLivArea_x_OverallQual'] = df['GrLivArea'] * df['OverallQual']
        df['TotalBsmtSF_x_BsmtQual'] = df['TotalBsmtSF'] * df['BsmtQual'].map(
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        )
        
        self.created_features.extend(['GrLivArea_x_OverallQual', 'TotalBsmtSF_x_BsmtQual'])
        return df
    
    def apply_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with all engineered features
        """
        df = self.create_total_sf(df)
        df = self.create_total_bathrooms(df)
        df = self.create_age_features(df)
        df = self.create_total_porch_sf(df)
        df = self.create_binary_features(df)
        df = self.create_quality_features(df)
        df = self.create_interaction_features(df)
        
        print(f"Created {len(self.created_features)} new features:")
        print(self.created_features)
        
        return df


if __name__ == "__main__":
    print("Feature engineering module ready")

