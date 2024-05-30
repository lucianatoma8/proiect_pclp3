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
numar_linii = data.shape[0]
linii_duplicate = data.duplicated().any()
if linii_duplicate == False:
    linii_duplicate = "Nu"
else:
    linii_duplicate = "Da"
# Examinarea structurii setului de date
print(f"\nNumarul de coloane: {numar_coloane}\n")
print(f"Tipurile datelor din fiecare coloana:\n{tipuri_date}\n")
print(f"Numarul de valori lipsa pentru fiecare coloana:\n{valori_lipsa}\n")
print(f"Numarul de linii: {numar_linii}\n")
print(f"Exista linii duplicate? {linii_duplicate}\n")

# CERINTA 2

# Procentul persoanelor care au supravietuit si care nu au supravietuit
rata_supravietuire = data['Survived'].value_counts(normalize=True) * 100
print(f"Procentul persoanelor care au supraviețuit: {rata_supravietuire[1]:.2f}%\n")
print(f"Procentul persoanelor care nu au supraviețuit: {rata_supravietuire[0]:.2f}%\n")

# Procentul pasagerilor pentru fiecare tip de clasa
procent_clasa = ((data['Pclass'].value_counts(normalize=True) * 100)).round(2)
print("Procentul pasagerilor pentru fiecare tip de clasa:\n\n")
for clasa, procent in zip(procent_clasa.index, procent_clasa.values):
    print(f" Clasa {clasa}: {procent}%")
print("\n")

# Procentul barbatilor si femeilor
procent_sex = data['Sex'].value_counts(normalize=True) * 100
print(f"Procentul barbatilor: {procent_sex['male']:.2f}%\n")
print(f"Procentul femeilor: {procent_sex['female']:.2f}%\n")

# Grafic pentru prezentarea rezultatelor
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(x=rata_supravietuire.index, y=rata_supravietuire.values, ax=ax[0], color='pink', alpha=0.8)
ax[0].set_title('Procentul de supravietuire')
ax[0].set_ylabel('Procent')
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['Nu', 'Da'])

sns.barplot(x=procent_clasa.index, y=procent_clasa.values, ax=ax[1], color='coral', alpha=0.8)
ax[1].set_title('Procentul pasagerilor pe clasa')
ax[1].set_ylabel('Procent')

sns.barplot(x=procent_sex.index, y=procent_sex.values, ax=ax[2], color='orange', alpha=0.8)
ax[2].set_title('Procentul pe sexe')
ax[2].set_ylabel('Procent')

plt.tight_layout()
plt.savefig("grafic1.png")
print(f"Graficul este salvat in grafic1.png\n")

# CERINTA 3

# Generarea histogramelor pentru coloanele numerice
coloane_numerice = data.select_dtypes(include=[np.number]).columns
data[coloane_numerice].hist(bins=30, figsize=(12, 10), layout=(3, 3), color=['navy'], alpha=0.75)
plt.suptitle('Histogramele pentru coloanele numerice')
plt.tight_layout()
plt.savefig("histograma.png")
print(f"Histograma este salvata in histograma.png\n")

# CERINTA 4

# Identificarea coloanelor cu valori lipsa
valori_lipsa = data.isnull().sum()
coloane_valori_lipsa = valori_lipsa[valori_lipsa > 0]
print("Coloanele pentru care exista valori lipsa:\n")
for coloana, valoare in coloane_valori_lipsa.items():
    print(f"{coloana}")
print('\n')

# Afisarea numarului si proportiei valorilor lipsa pentru fiecare coloana
print("Numarul si procentul valorilor lipsa pentru fiecare coloana:\n")
for coloana, valoare in zip(coloane_valori_lipsa.index, coloane_valori_lipsa.values):
    procent = (valoare / len(data)) * 100
    print(f"Coloana {coloana}: {valoare} valori lipsa, in proportie de {procent:.2f}%")
print("\n")

# Procentul pentru fiecare dintre cele 2 clase
print("Procentul valorilor lipsa pentru fiecare dintre cele doua clase (coloana Survived):\n") 
for coloana in coloane_valori_lipsa.index:
    for supravietuit in [0, 1]:
        procent = (data[coloana][data['Survived'] == supravietuit].isnull().sum() / len(data[data['Survived'] == supravietuit])) * 100
        print(f"Clasa {supravietuit}: {procent:.2f}% pentru coloana {coloana}")
print('\n')

# CERINTA 5

# Categoriile de varsta [0, 20], [21, 40], [41, 60], [61, max]
capete_intreval = [0, 20, 40, 60, data['Age'].max()]
categorii_varsta = ['0-20', '21-40', '41-60', '61+']

# Cati pasageri avem pentru fiecare categorie
# Numărarea pasagerilor pentru fiecare categorie de vârstă
nr_pasageri_per_categorie = pd.cut(data['Age'], bins=capete_intreval, right=True)
nr_pasageri_per_categorie = nr_pasageri_per_categorie.value_counts().sort_index()
nr_pasageri_per_categorie.index = categorii_varsta

print("Numărul de pasageri pentru fiecare categorie de vârstă:\n")
print(nr_pasageri_per_categorie)
print('\n')

# Introducerea unei coloane suplimentare cu indexul categoriei de varsta pentru fiecare exemplu
data['Index'] = pd.cut(data['Age'], bins=capete_intreval, labels=False, right=True)
index_categorii = [1, 2, 3, 4]
data['Index'] = data['Index'].map(dict(zip(range(len(categorii_varsta)), index_categorii)))

# Salvarea setului de date cu coloana suplimentara adaugata
data.to_csv("train1.csv", index=False)

print("Modificarile sunt vizibile in train1.csv")

# Crearea graficului
fig, ax = plt.subplots(figsize=(6, 5))

# Graficul pentru numărul de pasageri pentru fiecare categorie de vârstă
ax.bar(index_categorii, nr_pasageri_per_categorie, color='purple', alpha=0.75)
ax.set_title('Numarul de pasageri pentru fiecare categorie de varsta')
ax.set_xlabel('Indexul categoriei de varsta')
ax.set_ylabel('Numarul de pasageri')
ax.grid(axis='y')

plt.xticks(index_categorii)

# Salvarea graficului
plt.tight_layout()
plt.savefig("grafic2.png")
print("\nGraficul este salvat în grafic2.png\n")

# CERINTA 6

# Numarul de barbati care au supravietuit din fiecare categorie
supravietuitori_barbati = data[(data['Sex'] == 'male') & (data['Survived'] == 1)]
# nr_barbati_supravietuitori = len(supravietuitori_barbati)
# print(f"Numarul de barbati care au supravietuit: {nr_barbati_supravietuitori}")
supravietuitori_barbati_per_categorie = data[data['Sex'] == 'male'].groupby('Index')['Survived'].sum()
print('Numarul de barbati care au supravietuit pentru fiecare dintre cele 4 categorii de varsta:\n')
for index, numar in supravietuitori_barbati_per_categorie.items():
    print(f"Categoria {index}: {numar} barbati")
print('\n')

# Procentul de supravietuire al barbatilor in functie de categoria de varsta
total_barbati = data[data['Sex'] == 'male']
total_barbati_per_categorie = total_barbati['Index'].value_counts().sort_index()
procent_supravietuire_barbati = ((supravietuitori_barbati_per_categorie / total_barbati_per_categorie) * 100).round(2)
# for index, procent in procent_supravietuire_barbati.items():
#     print(f"Categoria {index}: {procent}%")
# print('\n')

# Graficul pentru influenta varstei asupra procentului de supravietuire a barbatilor
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(index_categorii, procent_supravietuire_barbati, color='red', alpha=0.75)
ax.set_title('Procentul de supravietuire al barbatilor in functie de categoria de varsta')
ax.set_xlabel('Indexul categoriei de varsta')
ax.set_ylabel('Procentul de supravietuire')
ax.grid(axis='y')
plt.xticks(index_categorii)
plt.tight_layout()
plt.savefig("grafic3.png")
print("Graficul este salvat in grafic3.png\n")

# CERINTA 7

# Aflarea procentului copiilor aflati la bord
copii = data[data['Age'] < 18]
procent_copii = (len(copii) / len(data)) * 100
print(f"Procentul copiilor aflati la bord: {procent_copii:.2f}%\n") 

adulti = data[data['Age'] >= 18]
# procent_adulti = (len(adulti) / len(data)) * 100
# print(f"Procentul adultilor aflati la bord: {procent_adulti:.2f}%\n") 

# Graficul pentru evidentierea ratei de supravietuire pentru copii si pentru adulti
fig, ax = plt.subplots(figsize=(6, 5))
rata_supravietuire_copii = copii['Survived'].value_counts(normalize=True) * 100
rata_supravietuire_adulti = adulti['Survived'].value_counts(normalize=True) * 100
# procent_copii_supravietuitori = rata_supravietuire_copii[1]
# procent_adulti_supravietuitori = rata_supravietuire_adulti[1]
# print(f"Procent copii supravietuitori: {procent_copii_supravietuitori.round(2)}%")
# print(f"Procent adulti supravietuitori: {procent_adulti_supravietuitori.round(2)}%")
# print('\n')
ax.bar(['Copii', 'Adulti'], [rata_supravietuire_copii[1], rata_supravietuire_adulti[1]], color=['darkseagreen', 'darkgreen'], alpha=0.8)
ax.set_title('Rata de supravietuire pentru copii si pentru adulti')
ax.set_ylabel('Procent')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig("grafic4.png")
print("Graficul este salvat in grafic4.png\n")

# CERINTA 8

# Consider coloanele categorice: Survived, PClass, Sex, Ticket, Cabin, Embarked, adica Pclass + cele de tip objet fara Name
# Crearea unei copii a setului de date pentru modificari
data_modificari = data.copy()

# Stim din CERINTELE 1/4 ca numai coloanele Age(float64), Cabin(object) si Embarked(object) au valori lipsa
# Coloanele categoriale sunt Cabin si Embarked

# Completarea valorilor lipsa pentru coloana 'Age' cu media pasagerilor din aceeasi clasa
data_modificari['Age'] = data_modificari.groupby('Survived')['Age'].transform(lambda x: x.fillna(x.mean()))

# Completarea valorilor lipsa pentru coloanele 'Cabin' si 'Embarked' cu cea mai frecventa valoare din aceeasi clasa
data_modificari['Cabin'] = data_modificari['Cabin'].fillna(data_modificari['Cabin'].mode()[0])
data_modificari['Embarked'] = data_modificari['Embarked'].fillna(data_modificari['Embarked'].mode()[0])

# data_aici = pd.read_csv('train1.csv')
# valori_lipsa_aici = data_aici.isnull().sum()
# coloane_valori_lipsa_aici = valori_lipsa_aici[valori_lipsa_aici > 0]
# print("Coloanele pentru care mai exista valori lipsa:\n")
# for coloana, valoare in coloane_valori_lipsa_aici.items():
#     print(f"{coloana}")
# print('\n')
# print("Numarul si procentul valorilor lipsa pentru fiecare coloana:\n")
# for coloana, valoare in zip(coloane_valori_lipsa_aici.index, coloane_valori_lipsa_aici.values):
#     procent = (valoare / len(data)) * 100
#     print(f"Coloana {coloana}: {valoare} valori lipsa, in proportie de {procent:.2f}%")
# print("\n")
# tipuri_date_index = data_aici.dtypes
# print(f"Tipurile datelor din coloana Index:\n{tipuri_date_index}\n")

# Completarea valorilor lipsa pentru coloana Index cu 0.0
data_modificari['Index'] = data_modificari['Index'].fillna(0.0)

# Salvarea setului de date cu modificarile adaugate
data_modificari.to_csv("train2.csv", index=False)

print("Modificarile sunt vizibile in train2.csv")

# CERINTA 9

# CERINTA 10
