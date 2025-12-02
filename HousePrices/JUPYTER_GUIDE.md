# 📓 Jupyter Notebooks - Rychlý návod

## 🚀 Instalace a spuštění

### 1. Nainstaluj Jupyter (pokud ještě není)

```bash
cd /Users/danielholub/SU/HousePrices
pip3 install jupyter ipykernel
```

Nebo nainstaluj všechny dependencies:
```bash
pip3 install -r requirements.txt
```

### 2. Spusť Jupyter server

```bash
cd /Users/danielholub/SU/HousePrices
jupyter notebook
```

Nebo pokud máš JupyterLab:
```bash
jupyter lab
```

**Co se stane:**
- Otevře se webový prohlížeč s Jupyter interface
- Uvidíš strukturu projektu
- Klikni na složku `notebooks/`

### 3. Otevři notebooky v pořadí

1. **`01_EDA.ipynb`** - Spusť všechny buňky (Shift+Enter nebo Cell → Run All)
2. **`02_Preprocessing.ipynb`** - Spusť všechny buňky
3. **`03_Modeling.ipynb`** - Postupně přidávej modely

---

## ⌨️ Základní klávesové zkratky

- **Shift + Enter** - Spustit buňku a přejít na další
- **Ctrl + Enter** - Spustit buňku (zůstaneš na ní)
- **Esc** - Edit mode → Command mode
- **A** - Přidat buňku nad (v command mode)
- **B** - Přidat buňku pod (v command mode)
- **DD** (2x D) - Smazat buňku (v command mode)
- **M** - Změnit na Markdown buňku (v command mode)
- **Y** - Změnit na Code buňku (v command mode)

---

## ✅ Jak překontrolovat, že vše funguje

### Kontrola 1: Jupyter běží
```bash
# V terminálu bys měl vidět něco jako:
# [I 2024-01-01 12:00:00.000 NotebookApp] Serving notebooks from local directory: /Users/danielholub/SU/HousePrices
# [I 2024-01-01 12:00:00.000 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888
```

### Kontrola 2: Notebook se načetl
- V Jupyter interface bys měl vidět soubory projektu
- Klikni na `notebooks/` → měl bys vidět `01_EDA.ipynb`, `02_Preprocessing.ipynb`, `03_Modeling.ipynb`

### Kontrola 3: Spuštění první buňky
1. Otevři `01_EDA.ipynb`
2. Klikni na první code buňku (s importy)
3. Stiskni **Shift + Enter**
4. Měl bys vidět výstup: `✅ Všechny knihovny načteny!`

### Kontrola 4: Kontrola výstupů
Po spuštění `02_Preprocessing.ipynb`:
```bash
# V terminálu zkontroluj, že se vytvořil soubor:
ls -lh data/processed_data.pkl
```

Měl bys vidět soubor s velikostí cca 5-10 MB.

### Kontrola 5: Načtení dat v modeling notebooku
1. Otevři `03_Modeling.ipynb`
2. Spusť buňku "Načtení preprocessovaných dat"
3. Měl bys vidět:
   ```
   ✅ Data načtena!
      X_train: (1458, XXX)
      X_test: (1459, XXX)
      y_train: (1458,)
   ```

---

## 🔧 Řešení problémů

### Problém: "ModuleNotFoundError"
**Řešení:**
```bash
pip3 install jupyter ipykernel pandas numpy matplotlib seaborn scikit-learn
```

### Problém: "Kernel not found"
**Řešení:**
```bash
python3 -m ipykernel install --user --name=python3
```

### Problém: Notebook se nespouští
**Řešení:**
1. Zkontroluj, že jsi v správném adresáři: `cd /Users/danielholub/SU/HousePrices`
2. Restartuj Jupyter server (Ctrl+C v terminálu, pak znovu `jupyter notebook`)

### Problém: Data se nenačítají
**Řešení:**
1. Zkontroluj, že existuje `data/processed_data.pkl`:
   ```bash
   ls -lh data/processed_data.pkl
   ```
2. Pokud neexistuje, spusť znovu `02_Preprocessing.ipynb`

---

## 📝 Tipy

1. **Vždy spouštěj buňky postupně** - ne "Run All" hned na začátku
2. **Kontroluj výstupy** - každá buňka by měla něco vytisknout
3. **Ukládej často** - Ctrl+S nebo File → Save
4. **Restartuj kernel** pokud něco nefunguje - Kernel → Restart

---

## 🎯 Doporučený workflow

1. **První běh:**
   - Otevři `01_EDA.ipynb`
   - Spusť všechny buňky (Cell → Run All)
   - Prohlédni si grafy a výsledky

2. **Preprocessing:**
   - Otevři `02_Preprocessing.ipynb`
   - Spusť všechny buňky
   - Zkontroluj, že se vytvořil `data/processed_data.pkl`

3. **Modelování:**
   - Otevři `03_Modeling.ipynb`
   - Spusť buňku "Načtení dat"
   - Postupně přidávej modely

---

**Hotovo!** Teď můžeš začít pracovat s notebooky! 🚀

