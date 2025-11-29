import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler #Pour le prétraitement

#Réglage optionnel d'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Style graphique
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('seaborn-v0_8')

#Charger les données
df=pd.read_csv('Mall_Customers.csv')

print(df.head())# Afficher premières lignes
print(df.info()) #info(types, non-null counts)
print(df.describe()) #Statistiques descriptives pour les colonnes non numériques
print(df.describe(include=['object'])) #Statistiques pour les colonnes non numériques

# Compter les valeurs manquantes par colonne
missing = df.isna().sum()
print("VALEURS MANQUANTES PAR COLONNE:\n", missing)

#Colonnes numériques usuelles : Age, Annual Income (k$), Spending Score (1-100)
num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

#Histogrammes
df[num_cols].hist(bins=15, figsize=(12,4))
plt.suptitle('Histogrammes des variables numériques')
plt.tight_layout()
plt.show()

#Boxplots pour repérer les outliers
plt.figure(figsize=(10,4))
sns.boxplot(data=df[num_cols])
plt.title('Boxplots des variables numériques')
plt.tight_layout()
plt.show()

#Sélection des features pour clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

#Compter les outliers par méthode IQR
def count_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

print("\nOutliers count:")
for col in features:
    print(f"{col}: {count_outliers_iqr(df[col])} outliers (IQR)")

#StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

#Assurer que les variables sont bien transformées
print("\nSTATISTIQUES APRÈS NORMALISATION:")
print(X_scaled.describe().T)

#Histogrammes
X_scaled.hist(bins=12, figsize=(10,3))
plt.suptitle('Histogrammes après normalisation')
plt.tight_layout()
plt.show()

#Conserver Customer ID & Gender si on veut les joindre plus tard;mais sauvegarder X_scaled pour clustering
preprocessed = df.copy()
preprocessed[features] = X_scaled[features]   # remplace les colonnes par leurs versions scalées

os.makedirs('results', exist_ok=True) # Créer le dossier 'results' s'il n'existe pas
#Sauvegarder
preprocessed.to_csv('results/mall_customers_preprocessed.csv', index=False)
print("DATA PRE"
      ""
      ""
      ""
      "TRAITÉE SAUVEGARDÉE/mall_customers_preprocessed.csv")