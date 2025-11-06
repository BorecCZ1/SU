import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Načti data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Základní přehled
print("Trénovací data:", train.shape)
print("Testovací data:", test.shape)

# Ukázka prvních 5 řádků
print(train.head())

# Info o typech dat
print(train.info())

# Základní popis číselných sloupců
print(train.describe())

# Ověř chybějící hodnoty
missing = train.isnull().sum()
print("Počet chybějících hodnot:\n", missing[missing > 0].sort_values(ascending=False))

# Rychlá vizualizace cílové proměnné (SalePrice)
sns.histplot(train["SalePrice"], kde=True)
plt.title("Distribuce prodejní ceny (SalePrice)")
plt.show()
