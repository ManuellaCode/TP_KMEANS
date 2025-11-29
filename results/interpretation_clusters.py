import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

#Chargement et préparation des données
df = pd.read_csv('../data/Mall_Customers.csv')
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Noms des clusters
cluster_names = {
    0: "Clients économes et prudents",
    1: "Jeunes dépensiers aisés",
    2: "Jeunes dépensiers modestes",
    3: "Clients moyens équilibrés",
    4: "Clients aisés mais prudents"
}
df['Cluster_Name'] = df['Cluster'].map(cluster_names)

# 1. STATISTIQUES DES CLUSTERS
print("="*60)
print("STATISTIQUES PAR CLUSTER")
print("="*60)
cluster_stats = df.groupby('Cluster_Name').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).round(1)
cluster_stats.columns = ['Âge moyen', 'Revenu moyen', 'Dépenses moyen', 'Nb clients']
print(cluster_stats)
# Segmentation Revenu vs Dépenses
plt.figure(figsize=(10, 6))
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'],
               cluster_data['Spending Score (1-100)'],
               label=cluster_names[cluster], alpha=0.7, s=50)
plt.xlabel('Revenu Annuel (k$)')
plt.ylabel('Score de Dépense')
plt.title('Segmentation: Revenu vs Dépenses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#  Répartition des clients par segment
plt.figure(figsize=(10, 6))
cluster_counts = df['Cluster_Name'].value_counts()
plt.bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette()[:5])
plt.title('Répartition des clients par segment')
plt.ylabel('Nombre de clients')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.show()
#  Profils comparés (normalisés)
plt.figure(figsize=(10, 6))
cluster_means = df.groupby('Cluster_Name')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
cluster_means_normalized = (cluster_means - cluster_means.mean()) / cluster_means.std()

x = np.arange(len(cluster_means))
width = 0.25
plt.bar(x - width, cluster_means_normalized['Age'], width, label='Âge', alpha=0.8)
plt.bar(x, cluster_means_normalized['Annual Income (k$)'], width, label='Revenu', alpha=0.8)
plt.bar(x + width, cluster_means_normalized['Spending Score (1-100)'], width, label='Dépenses', alpha=0.8)

plt.xticks(x, cluster_means.index, rotation=45)
plt.ylabel('Valeurs normalisées')
plt.title('Profils comparés des clusters (normalisés)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# GRAPHIQUE 4: Heatmap des caractéristiques
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Caractéristiques moyennes par cluster')
plt.tight_layout()
plt.show()
# 2. ANALYSE STRATÉGIQUE
print("\n" + "="*60)
print("ANALYSE STRATÉGIQUE")
print("="*60)

# Calcul de la valeur potentielle
segment_value = df.groupby('Cluster_Name').agg({
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
})
segment_value['Valeur_potentielle'] = (segment_value['Annual Income (k$)'] *
                                     segment_value['Spending Score (1-100)'] / 100)
segment_value = segment_value.sort_values('Valeur_potentielle', ascending=False)

for i, (segment, data) in enumerate(segment_value.iterrows(), 1):
    print(f"{i}. {segment}")
    print(f"   • Clients: {data['CustomerID']} ({data['CustomerID']/len(df)*100:.1f}%)")
    print(f"   • Revenu: {data['Annual Income (k$)']:.1f}k$ | Dépenses: {data['Spending Score (1-100)']:.1f}")
    print(f"   • Valeur: {data['Valeur_potentielle']:.1f}")
    # 3. RECOMMANDATIONS
    print("\n" + "=" * 60)
    print("RECOMMANDATIONS PAR SEGMENT")
    print("=" * 60)
    recommendations = {
        "Jeunes dépensiers aisés": "Cible premium - Produits exclusifs, programmes VIP",
        "Jeunes dépensiers modestes": "Croissance - Produits accessibles, marketing digital",
        "Clients aisés mais prudents": "Activation - Arguments rationnels, garanties",
        "Clients moyens équilibrés": "Fidélisation - Programmes standards, communications régulières",
        "Clients économes et prudents": "Optimisation - Offres basiques, focus rétention"
    }

    for segment, reco in recommendations.items():
        print(f"• {segment}: {reco}")