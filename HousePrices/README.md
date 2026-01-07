# 🏠 House Prices Prediction Project

Pokročilý projekt predikce cen nemovitostí pomocí machine learningu na Kaggle datasetu "House Prices: Advanced Regression Techniques".

## 🎯 Cíl projektu

Vytvořit komplexní ML pipeline pro predikci cen domů v Ames, Iowa, využitím různých regresních modelů a ensemble metod. Projekt zahrnuje:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multiple ML modely (od Linear Regression po Gradient Boosting)
- Ensemble metody (Stacking, Blending)
- Hyperparameter tuning
- Model interpretation (SHAP)

## 📊 Dataset

**Zdroj:** Kaggle - House Prices: Advanced Regression Techniques

**Data:**
- `train.csv`: 1460 domů s 80 features + SalePrice
- `test.csv`: 1459 domů bez SalePrice (pro predikci)
- `data_description.txt`: Detailní popis všech features

**Features zahrnují:**
- Rozměry domu (plochy, počet pokojů)
- Kvalitativní hodnocení (Overall Quality, Kitchen Quality, atd.)
- Lokalita (Neighborhood)
- Stáří a stav (Year Built, Overall Condition)
- Exteriér a interiér (Roof, Foundation, Basement)
- Venkovní prvky (Garage, Porch, Pool)

## 🚀 Quick Start

### Instalace dependencies

```bash
pip install -r requirements.txt
```

### Základní použití

```python
# TODO: Přidá se po implementaci
```

## 📁 Struktura projektu

```
HousePrices/
├── data/              # Data soubory
├── notebooks/         # Jupyter notebooks pro analýzu
├── src/              # Python moduly
├── models/           # Uložené modely
├── results/          # Výsledky a predikce
└── reports/          # Dokumentace a vizualizace
```

## 🔍 Metodologie

Viz `PROJECT_STRUCTURE.md` pro detailní plán projektu a strukturu dokumentace.

### Fáze projektu:

1. **EDA** - Pochopení dat a vztahů
2. **Preprocessing** - Čištění dat a handling missing values
3. **Feature Engineering** - Vytvoření nových features
4. **Modeling** - Trénování různých modelů
5. **Ensemble** - Kombinace modelů
6. **Optimization** - Hyperparameter tuning
7. **Evaluation** - Analýza výsledků a interpretace

## 🛠️ Použité technologie

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization:** Optuna
- **Interpretation:** SHAP
- **Development:** Jupyter, Python 3.10+

## 📈 Výsledky

TODO: Bude doplněno po dokončení analýzy

## 📝 Dokumentace

Kompletní dokumentace projektu (30-40 stran) zahrnuje:
- Teoretický základ ML metod
- Detailní EDA s vizualizacemi
- Feature engineering process
- Popis všech modelů a jejich výsledky
- Model interpretation
- Závěry a doporučení

## 👨‍💻 Autor

Daniel Holub - Školní projekt

## 📚 Reference

- [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Dataset: Dean De Cock (2011)

