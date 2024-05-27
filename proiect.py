import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurare stiluri pentru grafice
sns.set(style="darkgrid")

# CERINTA 1

# Citirea datelor din fisierul CSV
data = pd.read_csv('train.csv')

# Examinarea structurii setului de date
print(f"Numarul de coloane: {data.shape[1]}\n")
print(f"Tipurile datelor din fiecare coloana:\n{data.dtypes}\n")
print(f"Numarul de valori lipsa pentru fiecare coloana:\n{data.isnull().sum()}\n")
print(f"Exista linii duplicate? {data.duplicated().any()}\n")

# CERINTA 2

# Procentul persoanelor care au supravietuit si care nu au supravietuit
survival_rate = data['Survived'].value_counts(normalize=True) * 100
print(f"Procentul persoanelor care au supraviețuit: {survival_rate[1]:.2f}%\n")
print(f"Procentul persoanelor care nu au supraviețuit: {survival_rate[0]:.2f}%\n")

# Procentul pasagerilor pentru fiecare tip de clasa
class_rate = data['Pclass'].value_counts(normalize=True) * 100
print(f"Procentul pasagerilor pentru fiecare tip de clasa:\n{class_rate}\n")

# Procentul barbatilor si femeilor
gender_rate = df['Sex'].value_counts(normalize=True) * 100
print(f"Procentul barbatilor: {gender_rate['male']:.2f}%\n")
print(f"Procentul femeilor: {gender_rate['female']:.2f}%\n")

# Grafic pentru prezentarea rezultatelor
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x=survival_rate.index, y=survival_rate.values, ax=axs[0])
axs[0].set_title('Procentul de supravietuire')
axs[0].set_ylabel('Procent')
axs[0].set_xticklabels(['Nu', 'Da'])

sns.barplot(x=class_rate.index, y=class_rate.values, ax=axs[1])
axs[1].set_title('Procentul pasagerilor pe clasa')
axs[1].set_ylabel('Procent')

sns.barplot(x=gender_rate.index, y=gender_rate.values, ax=axs[2])
axs[2].set_title('Procentul pe sexe')
axs[2].set_ylabel('Procent')

plt.tight_layout()
plt.show()

# CERINTA 3

# CERINTA 4

# CERINTA 5

# CERINTA 6

# CERINTA 7

# CERINTA 8

# CERINTA 9

# CERINTA 10
