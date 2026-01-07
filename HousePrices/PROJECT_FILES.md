# 📁 Popis souborů v projektu

## 📓 Notebooky (hlavní práce)

### `notebooks/01_EDA.ipynb`
**Co dělá:** Exploratory Data Analysis - analýza dat
- Načítá data, zobrazuje statistiky
- Vytváří grafy (distribuce cen, korelace, missing values, outliers)
- Ukládá grafy do `results/`

### `notebooks/02_Preprocessing.ipynb`
**Co dělá:** Příprava dat pro modelování
- Handling missing values
- Feature Engineering (vytváří nové features)
- Encoding kategorických proměnných
- Odstranění outlierů
- **Ukládá preprocessovaná data do `data/processed_data.pkl`**

### `notebooks/03_Modeling.ipynb`
**Co dělá:** Trénování ML modelů
- Načítá preprocessovaná data z `data/processed_data.pkl`
- Trénuje různé modely (Linear, Ridge, XGBoost, atd.)
- Porovnává modely
- Vytváří finální predikce

---

## 📦 Moduly (`src/`)

### `src/data_loader.py`
**Co dělá:** Načítání dat
- `DataLoader` třída - načítá `train.csv` a `test.csv`
- Používá se v notebookách pro načtení originálních dat

### `src/feature_engineering.py`
**Co dělá:** Vytváření nových features
- `FeatureEngineer` třída - vytváří nové features (TotalSF, HouseAge, atd.)
- Používá se v `02_Preprocessing.ipynb`

### `src/models.py`
**Co dělá:** Wrapper pro ML modely
- `ModelTrainer` třída - trénuje různé modely
- Můžeš použít v `03_Modeling.ipynb` (volitelné)

### `src/evaluation.py`
**Co dělá:** Evaluace modelů
- `ModelEvaluator` třída - počítá metriky (RMSE, MAE, R²)
- Můžeš použít v `03_Modeling.ipynb` (volitelné)

### `src/utils.py`
**Co dělá:** Pomocné funkce
- `reduce_memory_usage()` - snižuje paměťovou náročnost
- Pomocné funkce pro vizualizace


---

## 📊 Data (`data/`)

### `data/train.csv`
**Co dělá:** Originální trénovací data (1460 domů, 81 sloupců)
- **NIKDY se nemění** - originální dataset

### `data/test.csv`
**Co dělá:** Test data pro predikci (1459 domů, 80 sloupců)
- **NIKDY se nemění** - originální dataset

### `data/data_description.txt`
**Co dělá:** Popis všech features v datasetu
- Dokumentace, co znamená každý sloupec

### `data/processed_data.pkl`
**Co dělá:** Preprocessovaná data (vytvoří se po spuštění `02_Preprocessing.ipynb`)
- Obsahuje `X_train`, `X_test`, `y_train`, `test_ids`
- Používá se v `03_Modeling.ipynb`

### `data/sample_submission.csv`
**Co dělá:** Ukázka formátu pro Kaggle submission
- Formát pro finální predikce

---

## 📈 Výsledky (`results/`)

### `results/*.png`
**Co dělá:** Grafy vytvořené v notebookách
- `01_saleprice_distribution.png` - distribuce cen
- `02_correlation_heatmap.png` - korelační matice
- `03_scatter_plots_top_features.png` - scatter plots
- `04_missing_values.png` - missing values
- `05_neighborhood_prices.png` - ceny podle lokality
- `06_quality_boxplot.png` - boxplot kvality
- `07_outliers.png` - detekce outlierů
- `08_model_comparison.png` - porovnání modelů

### `results/submission.csv`
**Co dělá:** Finální predikce pro Kaggle
- Vytvoří se po dokončení modelování

---

## 🤖 Modely (`models/`)

### `models/final_model.pkl`
**Co dělá:** Uložený natrénovaný model
- Vytvoří se po trénování v `03_Modeling.ipynb`
- Můžeš ho použít pro pozdější predikce

---

## 📝 Dokumentace

### `README.md`
**Co dělá:** Hlavní dokumentace projektu
- Popis projektu, struktura, technologie

### `JUPYTER_GUIDE.md`
**Co dělá:** Návod na použití Jupyter notebooků
- Jak nainstalovat a spustit Jupyter
- Základní klávesové zkratky
- Řešení problémů

### `PROJECT_STRUCTURE.md`
**Co dělá:** Plán projektu a struktura dokumentace
- Detailní plán pro 30-40 stran dokumentace
- Fáze projektu

### `PROJECT_FILES.md` (tento soubor)
**Co dělá:** Popis všech souborů v projektu

---

## ⚙️ Konfigurace

### `requirements.txt`
**Co dělá:** Seznam Python balíčků potřebných pro projekt
- Spusť `pip install -r requirements.txt` pro instalaci

---

## 📂 Prázdné složky

### `reports/figures/`
**Co dělá:** (Prázdná) - můžeš sem ukládat grafy pro dokumentaci

---

## ✅ Shrnutí workflow

1. **`01_EDA.ipynb`** → analyzuje data, vytváří grafy
2. **`02_Preprocessing.ipynb`** → připraví data, uloží do `processed_data.pkl`
3. **`03_Modeling.ipynb`** → načte `processed_data.pkl`, trénuje modely
4. Výsledky v `results/`, modely v `models/`

