"""
01 - EXPLORATORY DATA ANALYSIS (EDA)

Tento skript provede kompletní analýzu dat.
Pro dokumentaci ulož výstupy grafů!
"""

import sys
import os
# Přidej parent directory do path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data_loader import DataLoader
from src.utils import plot_missing_values, plot_correlation_heatmap

# Nastavení
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 80)
print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# ============================================================================
# 1. NAČTENÍ DAT
# ============================================================================
print("\n📂 1. Načítám data...")
# Najdi správnou cestu k data složce
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

loader = DataLoader(data_dir=data_dir)
train = loader.load_train_data()
test = loader.load_test_data()

# Zajisti results složku
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# ============================================================================
# 2. ZÁKLADNÍ PŘEHLED
# ============================================================================
print("\n📊 2. Základní přehled")
print("-" * 80)
print(f"   Celkem features: {train.shape[1]}")
print(f"   Numerické: {len(train.select_dtypes(include=[np.number]).columns)}")
print(f"   Kategorické: {len(train.select_dtypes(include=['object']).columns)}")
print(f"   Chybějící hodnoty: {train.isnull().sum().sum()}")

# ============================================================================
# 3. ANALÝZA SALEPRICE
# ============================================================================
print("\n💰 3. Analýza cílové proměnné (SalePrice)")
print("-" * 80)

print(f"   Mean:   ${train['SalePrice'].mean():>12,.0f}")
print(f"   Median: ${train['SalePrice'].median():>12,.0f}")
print(f"   Std:    ${train['SalePrice'].std():>12,.0f}")
print(f"   Min:    ${train['SalePrice'].min():>12,.0f}")
print(f"   Max:    ${train['SalePrice'].max():>12,.0f}")
print(f"\n   Skewness: {train['SalePrice'].skew():.3f}")
print(f"   Kurtosis: {train['SalePrice'].kurtosis():.3f}")

# Graf: Distribuce SalePrice
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(train['SalePrice'], bins=50, edgecolor='black', color='skyblue')
axes[0].set_xlabel('SalePrice ($)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of SalePrice', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# QQ plot
stats.probplot(train['SalePrice'], dist="norm", plot=axes[1])
axes[1].set_title('QQ Plot', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '01_saleprice_distribution.png'), dpi=150, bbox_inches='tight')
print("   ✅ Graf uložen: results/01_saleprice_distribution.png")
plt.close()

# Log transformace
print("\n   Po log transformaci:")
print(f"   Skewness: {np.log1p(train['SalePrice']).skew():.3f}")

# ============================================================================
# 4. KORELAČNÍ ANALÝZA
# ============================================================================
print("\n🔗 4. Korelační analýza")
print("-" * 80)

numeric_features = train.select_dtypes(include=[np.number]).columns
correlations = train[numeric_features].corr()['SalePrice'].sort_values(ascending=False)

print("\n   Top 15 pozitivních korelací:")
for i, (feature, corr) in enumerate(correlations[1:16].items(), 1):
    print(f"   {i:2d}. {feature:20s} : {corr:6.3f}")

# Heatmap korelací
plt.figure(figsize=(12, 10))
top_corr_features = correlations[1:16].index.tolist() + ['SalePrice']
corr_matrix = train[top_corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Top 15 Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
print("   ✅ Graf uložen: results/02_correlation_heatmap.png")
plt.close()

# ============================================================================
# 5. SCATTER PLOTS - TOP FEATURES
# ============================================================================
print("\n📈 5. Scatter plots pro top features")
print("-" * 80)

top_5_features = correlations[1:6].index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_5_features):
    axes[idx].scatter(train[feature], train['SalePrice'], alpha=0.5)
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('SalePrice', fontsize=10)
    axes[idx].set_title(f'{feature} vs SalePrice\nCorr: {correlations[feature]:.3f}', 
                       fontsize=11, fontweight='bold')
    axes[idx].grid(alpha=0.3)

fig.delaxes(axes[-1])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '03_scatter_plots_top_features.png'), dpi=150, bbox_inches='tight')
print("   ✅ Graf uložen: results/03_scatter_plots_top_features.png")
plt.close()

# ============================================================================
# 6. MISSING VALUES
# ============================================================================
print("\n❓ 6. Analýza chybějících hodnot")
print("-" * 80)

missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    print(f"\n   Celkem sloupců s missing values: {len(missing)}")
    print(f"   Celkem missing values: {missing.sum()}")
    print("\n   Top 10 sloupců:")
    for i, (col, count) in enumerate(missing.head(10).items(), 1):
        pct = (count / len(train)) * 100
        print(f"   {i:2d}. {col:20s} : {count:4d} ({pct:5.1f}%)")
    
    # Graf missing values
    plt.figure(figsize=(12, 6))
    missing.head(20).plot(kind='barh', color='salmon')
    plt.xlabel('Number of Missing Values', fontsize=12)
    plt.title('Missing Values by Feature (Top 20)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '04_missing_values.png'), dpi=150, bbox_inches='tight')
    print("   ✅ Graf uložen: results/04_missing_values.png")
    plt.close()
else:
    print("   ✅ Žádné chybějící hodnoty!")

# ============================================================================
# 7. KATEGORICKÉ PROMĚNNÉ
# ============================================================================
print("\n📦 7. Analýza kategorických proměnných")
print("-" * 80)

categorical_features = train.select_dtypes(include=['object']).columns

# Neighborhood
if 'Neighborhood' in train.columns:
    print("\n   Neighborhood - median ceny:")
    neighborhood_prices = train.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
    for i, (neighborhood, price) in enumerate(neighborhood_prices.head(10).items(), 1):
        print(f"   {i:2d}. {neighborhood:15s} : ${price:>10,.0f}")
    
    plt.figure(figsize=(12, 6))
    neighborhood_prices.plot(kind='bar', color='steelblue')
    plt.xlabel('Neighborhood', fontsize=12)
    plt.ylabel('Median SalePrice ($)', fontsize=12)
    plt.title('Median SalePrice by Neighborhood', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '05_neighborhood_prices.png'), dpi=150, bbox_inches='tight')
    print("   ✅ Graf uložen: results/05_neighborhood_prices.png")
    plt.close()

# OverallQual
if 'OverallQual' in train.columns:
    plt.figure(figsize=(12, 6))
    train.boxplot(column='SalePrice', by='OverallQual', figsize=(12, 6))
    plt.xlabel('Overall Quality (1-10)', fontsize=12)
    plt.ylabel('SalePrice ($)', fontsize=12)
    plt.title('SalePrice by Overall Quality', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '06_quality_boxplot.png'), dpi=150, bbox_inches='tight')
    print("   ✅ Graf uložen: results/06_quality_boxplot.png")
    plt.close()

# ============================================================================
# 8. OUTLIERS
# ============================================================================
print("\n🎯 8. Detekce outlierů")
print("-" * 80)

# GrLivArea outliers
if 'GrLivArea' in train.columns:
    Q1 = train['GrLivArea'].quantile(0.25)
    Q3 = train['GrLivArea'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = train[(train['GrLivArea'] < Q1 - 1.5 * IQR) | 
                     (train['GrLivArea'] > Q3 + 1.5 * IQR)]
    print(f"   GrLivArea outliers: {len(outliers)} domů")
    
    # Vizualizace
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].scatter(train['GrLivArea'], train['SalePrice'], alpha=0.5)
    axes[0].scatter(outliers['GrLivArea'], outliers['SalePrice'], 
                   color='red', s=100, alpha=0.7, label='Outliers')
    axes[0].set_xlabel('GrLivArea', fontsize=12)
    axes[0].set_ylabel('SalePrice', fontsize=12)
    axes[0].set_title('GrLivArea vs SalePrice - Outliers', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].boxplot(train['GrLivArea'])
    axes[1].set_ylabel('GrLivArea', fontsize=12)
    axes[1].set_title('GrLivArea Boxplot', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '07_outliers.png'), dpi=150, bbox_inches='tight')
    print("   ✅ Graf uložen: results/07_outliers.png")
    plt.close()

# ============================================================================
# 9. ZÁVĚRY PRO PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("📝 ZÁVĚRY PRO PREPROCESSING")
print("=" * 80)

print("\n1. SalePrice:")
print("   - Je výrazně skewed → aplikovat log transformaci")
print(f"   - Skewness před: {train['SalePrice'].skew():.3f}")
print(f"   - Skewness po log: {np.log1p(train['SalePrice']).skew():.3f}")

print("\n2. Missing Values:")
print(f"   - {len(missing)} sloupců obsahuje missing values")
print("   - Některé NA znamenají 'None' (např. PoolQC, Fence)")
print("   - Ostatní je třeba imputovat")

print("\n3. Nejdůležitější features:")
for i, feature in enumerate(top_5_features, 1):
    print(f"   {i}. {feature} (corr: {correlations[feature]:.3f})")

print("\n4. Outliers:")
print(f"   - GrLivArea má {len(outliers)} outlierů")
print("   - Zvážit odstranění extrémních hodnot")

print("\n5. Feature Engineering možnosti:")
print("   - TotalSF (kombinace ploch)")
print("   - Age features (HouseAge, RemodAge)")
print("   - Quality indices (OverallQual × OverallCond)")
print("   - Interaction features (GrLivArea × OverallQual)")

print("\n" + "=" * 80)
print("✅ EDA HOTOVO!")
print("=" * 80)
print(f"\nVšechny grafy uloženy v: results/")
print("Použij je pro dokumentaci!")
print("\nDalší krok: Spusť 02_preprocessing.py")

