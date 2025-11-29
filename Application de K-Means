# 3.1 MÉTHODE DU COUDE
inertia = [] 
silhouette_scores = []
k_range = range(2, 11)

# Calcul des modèles K-Means pour k = 2 à 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
plt.figure(figsize=(15, 5))

# Courbe d'inertie
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8, markerfacecolor='red')
plt.title('Méthode du Coude - Inertie', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

# Courbe des scores de silhouette
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8, markerfacecolor='blue')
plt.title('Score de Silhouette', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

plt.tight_layout()
plt.savefig("methode_du_coude.png")
plt.show()

# Entraîner le modèle final avec le k choisi
optimal_k = k_range[np.argmax(silhouette_scores)]
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)
df['Cluster'] = kmeans_final.labels_

# Visualisation des clusters
colors = plt.get_cmap("Set3")(np.linspace(0, 1, optimal_k))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    axes[0].scatter(cluster_data['Age'], cluster_data['Spending_Score_(1-100)'],
                    c=[colors[cluster]], label=f'Cluster {cluster}', alpha=0.8, s=60)
axes[0].set_title('Clusters : Âge vs Score de Dépense', fontweight='bold')
axes[0].legend()

for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    axes[1].scatter(cluster_data['Annual_Income_(k$)'], cluster_data['Spending_Score_(1-100)'],
                    c=[colors[cluster]], label=f'Cluster {cluster}', alpha=0.8, s=60)
axes[1].set_title('Clusters : Revenu vs Score de Dépense', fontweight='bold')
axes[1].legend()

for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    axes[2].scatter(cluster_data['Age'], cluster_data['Annual_Income_(k$)'],
                    c=[colors[cluster]], label=f'Cluster {cluster}', alpha=0.8, s=60)
axes[2].set_title('Clusters : Âge vs Revenu', fontweight='bold')
axes[2].legend()
plt.tight_layout()
plt.show()
