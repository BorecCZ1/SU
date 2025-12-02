# 📁 Struktura projektu - Stručný přehled

## 🎯 Hlavní soubory (co potřebuješ)

### 📓 Notebooky
- **`notebooks/01_EDA.ipynb`** - Analýza dat (grafy, statistiky)
- **`notebooks/02_Preprocessing.ipynb`** - Příprava dat (vytvoří `processed_data.pkl`)
- **`notebooks/03_Modeling.ipynb`** - Trénování modelů

### 📦 Moduly (`src/`)
- **`data_loader.py`** - Načítá `train.csv` a `test.csv`
- **`feature_engineering.py`** - Vytváří nové features (TotalSF, HouseAge, atd.)
- **`models.py`** - Wrapper pro ML modely (volitelné)
- **`evaluation.py`** - Evaluace modelů (volitelné)
- **`utils.py`** - Pomocné funkce

### 📊 Data (`data/`)
- **`train.csv`** - Originální trénovací data (1460 domů)
- **`test.csv`** - Test data (1459 domů)
- **`data_description.txt`** - Popis všech features
- **`processed_data.pkl`** - Preprocessovaná data (vytvoří se po spuštění `02_Preprocessing.ipynb`)

### 📈 Výsledky (`results/`)
- **`*.png`** - Grafy z EDA a modelování
- **`submission.csv`** - Finální predikce pro Kaggle

### 🤖 Modely (`models/`)
- **`final_model.pkl`** - Uložený natrénovaný model

### 📝 Dokumentace
- **`README.md`** - Hlavní popis projektu
- **`JUPYTER_GUIDE.md`** - Návod na použití Jupyter
- **`PROJECT_FILES.md`** - Detailní popis všech souborů
- **`PROJECT_STRUCTURE.md`** - Plán dokumentace (30-40 stran)

### ⚙️ Konfigurace
- **`requirements.txt`** - Python balíčky (`pip install -r requirements.txt`)

---

## ✅ Workflow

1. **`01_EDA.ipynb`** → analyzuje data, vytváří grafy
2. **`02_Preprocessing.ipynb`** → připraví data, uloží do `processed_data.pkl`
3. **`03_Modeling.ipynb`** → načte `processed_data.pkl`, trénuje modely

---

**Více detailů:** Viz `PROJECT_FILES.md`

