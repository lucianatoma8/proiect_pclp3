import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CERINTA 1

# Citirea datelor din fisierul CSV
data = pd.read_csv('train.csv')

numar_coloane = len(data.columns)
tipuri_date = data.dtypes
valori_lipsa = data.isnull().sum()
linii_duplicate = data.duplicated().any()
if linii_duplicate == False:
    linii_duplicate = "Nu"
else:
    linii_duplicate = "Da"
# Examinarea structurii setului de date
print(f"\nNumarul de coloane: {numar_coloane}\n")
print(f"Tipurile datelor din fiecare coloana:\n{tipuri_date}\n")
print(f"Numarul de valori lipsa pentru fiecare coloana:\n{valori_lipsa}\n")
print(f"Exista linii duplicate? {linii_duplicate}\n")

