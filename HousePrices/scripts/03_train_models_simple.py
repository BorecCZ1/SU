"""
03 - TRAIN MODELS (SIMPLE VERSION)

Verze bez XGBoost/LightGBM - použije jen scikit-learn modely.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

print("=" * 80)
print("🤖 MODEL TRAINING & EVALUATION (Simple Version)")
print("=" * 80)

# ============================================================================
# 1. NAČTENÍ PREPROCESSOVANÝCH DAT
# ============================================================================
print("\n📂 1. Načítám preprocessovaná data...")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
models_dir = os.path.join(project_dir, 'models')

with open(os.path.join(data_dir, 'processed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
test_ids = data['test_ids']

# Nahraď případné zbylé NaN hodnoty
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}")

# ============================================================================
# 2. BASELINE MODELY
# ============================================================================
print("\n🎯 2. Trénování baseline modelů...")
print("-" * 80)

# Připrav cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=10.0),
    'Lasso': Lasso(alpha=0.001, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
}

results = {}

for name, model in models.items():
    print(f"\n   Trénuji {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        scoring='neg_root_mean_squared_error',
        cv=kfold,
        n_jobs=-1
    )
    rmse = -cv_scores.mean()
    std = cv_scores.std()
    
    results[name] = {
        'RMSE': rmse,
        'Std': std
    }
    
    print(f"   RMSE: {rmse:.4f} (+/- {std:.4f})")

print("\n   ✅ Baseline modely hotové")

# ============================================================================
# 3. TREE-BASED MODELY (scikit-learn only)
# ============================================================================
print("\n🌲 3. Trénování tree-based modelů...")
print("-" * 80)

tree_models = {
    'DecisionTree': DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
}

for name, model in tree_models.items():
    print(f"\n   Trénuji {name}...")
    
    cv_scores = cross_val_score(
        model, X_train, y_train,
        scoring='neg_root_mean_squared_error',
        cv=kfold,
        n_jobs=-1
    )
    rmse = -cv_scores.mean()
    std = cv_scores.std()
    
    results[name] = {
        'RMSE': rmse,
        'Std': std
    }
    
    print(f"   RMSE: {rmse:.4f} (+/- {std:.4f})")

print("\n   ✅ Tree-based modely hotové")

# ============================================================================
# 4. POROVNÁNÍ MODELŮ
# ============================================================================
print("\n📊 4. Porovnání všech modelů")
print("-" * 80)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('RMSE')

print("\n" + results_df.to_string())

# Vizualizace
plt.figure(figsize=(12, 6))
models_names = results_df.index
rmse_values = results_df['RMSE']
std_values = results_df['Std']

bars = plt.barh(models_names, rmse_values, color='steelblue', alpha=0.8)
plt.errorbar(rmse_values, models_names, xerr=std_values, 
             fmt='none', color='red', capsize=5, alpha=0.7)

# Obarvi nejlepší model
bars[0].set_color('green')
bars[0].set_alpha(1.0)

plt.xlabel('RMSE (Cross-Validation)', fontsize=12)
plt.title('Model Comparison - Cross-Validation RMSE', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '08_model_comparison.png'), dpi=150, bbox_inches='tight')
print("\n   ✅ Graf uložen: results/08_model_comparison.png")
plt.close()

# ============================================================================
# 5. NATRÉNUJ NEJLEPŠÍ MODEL NA CELÝCH DATECH
# ============================================================================
print("\n🏆 5. Trénuji nejlepší model na celých datech...")
print("-" * 80)

best_model_name = results_df.index[0]
print(f"   Nejlepší model: {best_model_name}")
print(f"   RMSE: {results_df.iloc[0]['RMSE']:.4f}")

# Natrénuj finální model
final_model = tree_models.get(best_model_name, models.get(best_model_name))
final_model.fit(X_train, y_train)

print("   ✅ Finální model natrénovaný")

# Feature importance (pokud je k dispozici)
if hasattr(final_model, 'feature_importances_'):
    print("\n   Top 15 nejdůležitějších features:")
    
    importances = final_model.feature_importances_
    feature_names = X_train.columns
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))
    
    # Graf
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '09_feature_importance.png'), dpi=150, bbox_inches='tight')
    print("   ✅ Graf uložen: results/09_feature_importance.png")
    plt.close()

# ============================================================================
# 6. PREDIKCE NA TESTOVACÍCH DATECH
# ============================================================================
print("\n🎯 6. Predikce na testovacích datech...")

predictions_log = final_model.predict(X_test)
predictions = np.expm1(predictions_log)  # Převeď z log zpět

print(f"   Predikovaných cen: {len(predictions)}")
print(f"   Průměrná predikovaná cena: ${predictions.mean():,.0f}")
print(f"   Min: ${predictions.min():,.0f}")
print(f"   Max: ${predictions.max():,.0f}")

# ============================================================================
# 7. VYTVOŘENÍ SUBMISSION SOUBORU
# ============================================================================
print("\n💾 7. Vytvářím submission soubor...")

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

submission.to_csv(os.path.join(results_dir, 'submission.csv'), index=False)
print("   ✅ Submission uložen: results/submission.csv")

# ============================================================================
# 8. ULOŽENÍ MODELU
# ============================================================================
print("\n💾 8. Ukládám model...")

with open(os.path.join(models_dir, 'final_model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)

print("   ✅ Model uložen: models/final_model.pkl")

# ============================================================================
# ZÁVĚR
# ============================================================================
print("\n" + "=" * 80)
print("🎉 MODEL TRAINING HOTOVO!")
print("=" * 80)

print(f"\n✅ Nejlepší model: {best_model_name}")
print(f"✅ Cross-validation RMSE: {results_df.iloc[0]['RMSE']:.4f}")
print(f"✅ Submission vytvořen: results/submission.csv")
print(f"✅ Model uložen: models/final_model.pkl")

print("\n📊 Pro dokumentaci použij:")
print("   - results/08_model_comparison.png")
print("   - results/09_feature_importance.png")
print("   - Tabulku výsledků (výše)")

print("\n💡 Tip: Pokud chceš použít XGBoost/LightGBM:")
print("   brew install libomp")
print("   Pak můžeš použít 03_train_models.py")

print("\n" + "=" * 80)

