# 📊 Segmentation Client - Analyse K-Means

## 📋 Description
Projet d'analyse de segmentation client utilisant l'algorithme K-Means pour identifier des groupes de clients homogènes basés sur leurs caractéristiques socio-économiques et comportementales.

## 🎯 Objectif
Identifier des segments clients distincts pour élaborer des stratégies marketing ciblées et optimiser l'expérience client.

## 📁 Structure du Projet
TP K-MEANS/
│
├── 📊 Mall_Customers.csv # Données clients
├── 🐍 interpretation_clusters.py # Script d'analyse
├── 📄 README.md # Documentation
└── 📁 .idea/ # Configuration PyCharm

## 🛠️ Technologies Utilisées
- **Python 3**
- **Scikit-learn** (K-Means, StandardScaler)
- **Pandas & NumPy**
- **Matplotlib & Seaborn**

## 📈 Résultats Clés
L'analyse a identifié **5 segments clients** :

1. **🎯 Jeunes dépensiers aisés** (17.5%) - Cible prioritaire
2. **🚀 Jeunes dépensiers modestes** (11.5%) - Potentiel croissance  
3. **💎 Clients aisés mais prudents** (40.5%) - Plus grand segment
4. **⚖️ Clients moyens équilibrés** (11%) - Base fidèle
5. **🛡️ Clients économes et prudents** (19.5%) - Optimisation coûts

## 🚀 Installation et Utilisation

```bash
# Cloner le repository
git clone https://github.com/Valencia-glitch/tp-kmeans.git

# Se positionner dans le dossier
cd "TP K-MEANS"

# Exécuter l'analyse
python interpretation_clusters.py

📊 Visualisations Générées

    Graphique Revenu vs Dépenses par segment

    Répartition des clients par cluster

    Profils comparés des segments (normalisés)

    Heatmap des caractéristiques moyennes
