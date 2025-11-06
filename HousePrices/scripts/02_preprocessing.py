"""
02 - DATA PREPROCESSING

Čištění dat a příprava pro modelování.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

print("=" * 80)
print("🧹 DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 80)

# ============================================================================
# 1. NAČTENÍ DAT
# ============================================================================
print("\n📂 1. Načítám data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

loader = DataLoader(data_dir=data_dir)
train = loader.load_train_data()
test = loader.load_test_data()

print(f"   Train shape: {train.shape}")
print(f"   Test shape: {test.shape}")

# Ulož test IDs pro později (před jakoukoliv filtrací)
test_ids_original = test['Id'].copy()

# ============================================================================
# 2. LOG TRANSFORMACE SALEPRICE
# ============================================================================
print("\n📊 2. Log transformace SalePrice...")
y_train = np.log1p(train['SalePrice'])
print(f"   Original skewness: {train['SalePrice'].skew():.3f}")
print(f"   After log skewness: {y_train.skew():.3f}")
print("   ✅ Transformace aplikována")

# Odstraň SalePrice z train (budeme používat y_train)
train = train.drop('SalePrice', axis=1)

# ============================================================================
# 3. SPOJENÍ TRAIN A TEST PRO KONZISTENTNÍ PREPROCESSING
# ============================================================================
print("\n🔗 3. Spojuji train a test pro konzistentní preprocessing...")
n_train = train.shape[0]
all_data = pd.concat([train, test], axis=0, sort=False)
print(f"   Combined shape: {all_data.shape}")

# ============================================================================
# 4. HANDLING MISSING VALUES
# ============================================================================
print("\n❓ 4. Řešení chybějících hodnot...")

# NA znamená "None" pro tyto features
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType', 'MSSubClass']

for col in none_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

# Numerické - nahraď 0
zero_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
             'GarageYrBlt', 'GarageArea', 'GarageCars']

for col in zero_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# LotFrontage - nahraď median podle Neighborhood
if 'LotFrontage' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

# Ostatní numerické - median
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Ostatní kategorické - mode
categorical_cols = all_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

print(f"   Zbývající missing values: {all_data.isnull().sum().sum()}")
print("   ✅ Missing values vyřešeny")

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================
print("\n🔧 5. Feature Engineering...")

fe = FeatureEngineer()
all_data = fe.apply_all_features(all_data)

print(f"   ✅ Vytvořeno {len(fe.created_features)} nových features")

# ============================================================================
# 6. ODSTRANĚNÍ OUTLIERŮ (pouze z train)
# ============================================================================
print("\n🎯 6. Odstranění outlierů...")

# Identifikuj outliery v GrLivArea (pouze v train části)
train_data_temp = all_data[:n_train].copy()
train_data_temp['SalePrice'] = y_train.values

# Odstranění domů s GrLivArea > 4000 a nízkou cenou
outliers_idx = train_data_temp[(train_data_temp['GrLivArea'] > 4000) & 
                               (train_data_temp['SalePrice'] < 12.5)].index

if len(outliers_idx) > 0:
    print(f"   Odstraněno {len(outliers_idx)} outlierů")
    # Odstraň z train data
    all_data = all_data.drop(outliers_idx)
    y_train = y_train.drop(outliers_idx)
    n_train = n_train - len(outliers_idx)
else:
    print("   Žádné outliery k odstranění")

# ============================================================================
# 7. ENCODING KATEGORICKÝCH PROMĚNNÝCH
# ============================================================================
print("\n🔤 7. Encoding kategorických proměnných...")

# Label encoding pro ordinální proměnné
ordinal_map = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
}

for col, mapping in ordinal_map.items():
    if col in all_data.columns:
        all_data[col] = all_data[col].map(mapping)

# One-hot encoding pro ostatní kategorické
categorical_features = all_data.select_dtypes(include=['object']).columns
all_data = pd.get_dummies(all_data, columns=categorical_features, drop_first=True)

print(f"   Features po encoding: {all_data.shape[1]}")
print("   ✅ Encoding hotový")

# ============================================================================
# 8. ROZDĚLENÍ ZPĚT NA TRAIN A TEST
# ============================================================================
print("\n✂️  8. Rozdělení zpět na train a test...")

X_train = all_data[:n_train]
X_test = all_data[n_train:]

# Vyber správné test IDs (některé mohly být odstraněny během preprocessingu)
test_ids = test_ids_original.iloc[X_test.index - n_train].reset_index(drop=True)

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   test_ids shape: {test_ids.shape}")

# ============================================================================
# 9. ULOŽENÍ PREPROCESSOVANÝCH DAT
# ============================================================================
print("\n💾 9. Ukládám preprocessovaná data...")

# Ulož jako pickle pro rychlé načítání
with open(os.path.join(data_dir, 'processed_data.pkl'), 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'test_ids': test_ids,
        'feature_names': X_train.columns.tolist()
    }, f)

print("   ✅ Data uložena: data/processed_data.pkl")

# ============================================================================
# 10. ZÁVĚREČNÉ STATISTIKY
# ============================================================================
print("\n" + "=" * 80)
print("📊 ZÁVĚREČNÉ STATISTIKY")
print("=" * 80)

print(f"\nTrain data:")
print(f"   Samples: {X_train.shape[0]}")
print(f"   Features: {X_train.shape[1]}")

print(f"\nTest data:")
print(f"   Samples: {X_test.shape[0]}")
print(f"   Features: {X_test.shape[1]}")

print(f"\nTarget (log SalePrice):")
print(f"   Mean: {y_train.mean():.3f}")
print(f"   Std: {y_train.std():.3f}")
print(f"   Min: {y_train.min():.3f}")
print(f"   Max: {y_train.max():.3f}")

print("\n" + "=" * 80)
print("✅ PREPROCESSING HOTOVO!")
print("=" * 80)
print("\nData připravena pro modelování!")
print("Další krok: Spusť 03_baseline_models.py")

