# 🏠 House Prices Prediction - Struktura Projektu

## 📋 Obsah pro dokumentaci (30-40 stran)

### **1. ÚVOD (2-3 strany)**
- Popis problému a cíle projektu
- Motivace - proč je predikce cen nemovitostí důležitá
- Přehled použitých metod a technologií
- Struktura práce

### **2. TEORETICKÝ ZÁKLAD (4-5 stran)**
- **Regresní analýza**
  - Lineární regrese
  - Ridge a Lasso regrese
  - Elastic Net
- **Ensemble metody**
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - Stacking a Blending
- **Metriky hodnocení**
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² score
  - Cross-validation

### **3. EXPLORATORY DATA ANALYSIS (EDA) (6-8 stran)**
#### 3.1 Základní přehled dat
- Počet záznamů a features
- Datové typy
- Chybějící hodnoty (missing values)

#### 3.2 Analýza cílové proměnné (SalePrice)
- Distribuce cen
- Statistické charakteristiky (mean, median, std)
- Detekce outlierů
- Transformace (log transform pro normalizaci)

#### 3.3 Numerické proměnné
- Korelační analýza
- Vztah k ceně (scatter plots)
- Top 10 nejvlivnějších features

#### 3.4 Kategorické proměnné
- Analýza kategorií
- Box plots pro jednotlivé kategorie vs. cena
- Identifikace důležitých kategorií

#### 3.5 Multivariantní analýza
- Heatmap korelací
- Pair plots pro top features
- Interakce mezi proměnnými

### **4. DATA PREPROCESSING (5-6 stran)**
#### 4.1 Řešení chybějících hodnot
- Strategie pro numerické proměnné (median/mean imputation)
- Strategie pro kategorické (mode/None)
- Speciální případy (LotFrontage, GarageYrBlt)

#### 4.2 Feature Engineering
- **Vytvoření nových features:**
  - TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
  - TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
  - HouseAge = YrSold - YearBuilt
  - RemodAge = YrSold - YearRemodAdd
  - TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
  - HasPool, HasGarage, HasBasement (binary features)
  - QualityIndex = OverallQual * OverallCond
- **Polynomial features** pro důležité proměnné
- **Interaction features** (např. GrLivArea * OverallQual)

#### 4.3 Encoding kategorických proměnných
- Label Encoding
- One-Hot Encoding
- Target Encoding (mean encoding)

#### 4.4 Scaling a normalizace
- StandardScaler
- RobustScaler (lepší pro outliers)
- Log transformace pro skewed features

#### 4.5 Odstranění outlierů
- Z-score metoda
- IQR metoda
- Vliv na model

### **5. MODELOVÁNÍ (8-10 stran)**
#### 5.1 Baseline modely
- **Linear Regression**
  - Implementace
  - Výsledky
  - Analýza residuí
- **Ridge Regression**
  - Alpha tuning
  - Regularizace
- **Lasso Regression**
  - Feature selection
  - Optimální alpha

#### 5.2 Pokročilé modely
- **Elastic Net**
  - Kombinace L1 a L2 regularizace
- **Decision Trees**
  - Základní implementace
  - Overfitting problém
- **Random Forest**
  - Hyperparametry (n_estimators, max_depth, min_samples_split)
  - Feature importance
- **Gradient Boosting modely**
  - **XGBoost**
    - Learning rate, n_estimators
    - Tree-specific params
  - **LightGBM**
    - Rychlost vs. přesnost
    - Handling kategorických proměnných
  - **CatBoost**
    - Built-in categorical handling
  - **Porovnání gradient boosting modelů**

#### 5.3 Ensemble metody
- **Voting Regressor**
  - Hard vs. Soft voting
- **Stacking**
  - Multiple base models
  - Meta-learner
- **Blending**
  - Weighted average
  - Optimální váhy

### **6. HYPERPARAMETER TUNING (3-4 strany)**
- **Grid Search**
  - Exhaustive search
  - Výhody a nevýhody
- **Random Search**
  - Efektivnější přístup
- **Bayesian Optimization (Optuna)**
  - Inteligentní hledání
  - Trials a pruning
- **Cross-validation strategie**
  - K-Fold
  - Stratified K-Fold

### **7. MODEL EVALUATION & INTERPRETATION (4-5 stran)**
#### 7.1 Metriky
- RMSE na train/validation/test
- MAE
- R² score
- Comparison table všech modelů

#### 7.2 Feature Importance
- Tree-based importance
- Permutation importance
- SHAP values
  - Summary plots
  - Dependence plots
  - Force plots

#### 7.3 Error Analysis
- Residual plots
- Predikce vs. skutečnost
- Případy s největší chybou
- Kde model selhává

### **8. VÝSLEDKY A PREDIKCE (2-3 strany)**
- Finální model
- Predikce na test.csv
- Submission file pro Kaggle
- Srovnání s Kaggle leaderboard (pokud submituješ)

### **9. ZÁVĚR (2-3 strany)**
- Shrnutí výsledků
- Co fungovalo nejlépe
- Možná vylepšení
- Praktické aplikace
- Lessons learned

### **10. PŘÍLOHY**
- Kompletní kód
- Další grafy a tabulky
- Literatura a zdroje

---

## 📁 Doporučená struktura souborů

```
HousePrices/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb          # Data cleaning & feature engineering
│   ├── 03_baseline_models.ipynb        # Linear, Ridge, Lasso
│   ├── 04_advanced_models.ipynb        # RF, XGBoost, LightGBM, CatBoost
│   ├── 05_ensemble.ipynb               # Stacking & Blending
│   ├── 06_hyperparameter_tuning.ipynb  # Optimization
│   └── 07_final_predictions.ipynb      # Test predictions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Načítání dat
│   ├── preprocessing.py        # Preprocessing funkce
│   ├── feature_engineering.py  # Feature engineering
│   ├── models.py              # Model wrappers
│   ├── evaluation.py          # Metriky a evaluace
│   └── utils.py               # Helper funkce
│
├── models/
│   ├── ridge_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   └── final_ensemble.pkl
│
├── results/
│   ├── submission.csv
│   ├── model_comparison.csv
│   └── feature_importance.png
│
├── reports/
│   └── figures/               # Grafy pro dokumentaci
│
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

---

## 🛠️ Co budeš dělat krok za krokem

### **FÁZE 1: EDA & Understanding (2-3 dny)**
1. Načti data a prozkoumej strukturu
2. Analýza missing values
3. Statistická analýza všech proměnných
4. Vizualizace distribucí
5. Korelační analýza
6. Identifikace outlierů

**Output pro dokumentaci:**
- Tabulky se statistikami
- Grafy distribucí
- Heatmapy korelací
- Seznamy missing values

### **FÁZE 2: Preprocessing & Feature Engineering (2-3 dny)**
1. Handling missing values
2. Vytvoření nových features
3. Encoding kategorických proměnných
4. Scaling numerických proměnných
5. Log transformace skewed features
6. Odstranění outlierů
7. Split train/validation

**Output pro dokumentaci:**
- Před/po statistiky
- Seznam nových features
- Zdůvodnění jednotlivých kroků

### **FÁZE 3: Baseline Models (1-2 dny)**
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Porovnání výsledků

**Output pro dokumentaci:**
- Comparison table
- Residual plots
- Feature coefficients

### **FÁZE 4: Advanced Models (3-4 dny)**
1. Random Forest
2. XGBoost
3. LightGBM
4. CatBoost
5. Feature importance analysis

**Output pro dokumentaci:**
- Performance metriky
- Feature importance plots
- Training curves

### **FÁZE 5: Ensemble & Optimization (2-3 dny)**
1. Stacking různých modelů
2. Blending
3. Hyperparameter tuning (Optuna)
4. Cross-validation
5. Final model selection

**Output pro dokumentaci:**
- Ensemble architecture diagram
- Tuning results
- CV scores

### **FÁZE 6: Evaluation & Interpretation (1-2 dny)**
1. SHAP analysis
2. Error analysis
3. Model comparison
4. Final predictions

**Output pro dokumentaci:**
- SHAP plots
- Error distributions
- Final comparison table

### **FÁZE 7: Documentation (3-5 dní)**
1. Sepiš teoretickou část
2. Přidej výsledky z notebooků
3. Vytvoř profesionální grafy
4. Napiš závěry

---

## 💡 Tipy pro rozsáhlou dokumentaci

### **Jak dosáhnout 30-40 stran:**

1. **Podrobné popisy metod** (4-5 stran)
   - Matematické vzorce
   - Vysvětlení algoritmů
   - Výhody a nevýhody

2. **Mnoho vizualizací** (15-20 grafů)
   - Každý graf zabere 0.5-1 stranu
   - Vždy s popisem a interpretací

3. **Tabulky s výsledky** (10-15 tabulek)
   - Statistiky
   - Model comparison
   - Hyperparameters

4. **Code snippets** (důležité části kódu)
   - Feature engineering funkce
   - Model implementace
   - S komentáři

5. **Detailní analýza výsledků**
   - Pro každý model samostatná sekce
   - Co funguje, co ne
   - Proč

6. **Feature descriptions**
   - Popište všechny důležité features
   - Jejich vliv na cenu

---

## 🎓 Co se naučíš

- **Data Science workflow** od A do Z
- **Feature engineering** - nejdůležitější skill
- **Multiple ML algorithms** a kdy jaký použít
- **Ensemble methods** - kombinace modelů
- **Hyperparameter tuning** - optimalizace
- **Model interpretation** - SHAP, feature importance
- **Production thinking** - code organization

---

## 📊 Očekávané výsledky

**Dobré RMSE pro tento dataset:** ~0.12-0.13 (po log transformaci ceny)

**Top features obvykle jsou:**
- OverallQual
- GrLivArea
- TotalBsmtSF
- GarageCars
- YearBuilt
- Neighborhood (některé)

**Best models obvykle:**
1. XGBoost / LightGBM (single model)
2. Stacked ensemble (multiple models)

---

Chceš, abych ti vytvořil i kostry těch jednotlivých notebooků nebo rovnou začneme s EDA?

