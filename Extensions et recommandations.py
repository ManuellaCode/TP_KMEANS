# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 8))
for cluster in range(optimal_k):
    cluster_data = X_pca[df['Cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                c=[colors[cluster]], label=f'Cluster {cluster}', alpha=0.8, s=60)
plt.title('Visualisation des Clusters avec PCA', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
plt.figure(figsize=(10, 8))
for label in set(labels_dbscan):
    cluster_data = X_pca[np.array(labels_dbscan) == label]
    cmap = plt.get_cmap('tab10')  # récupère la colormap tab10
    color = 'black' if label == -1 else cmap(label % 10)  # %10 pour être sûr
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color,
                label=f'Cluster {label}', alpha=0.7, s=60)
plt.title("Clusters obtenus avec DBSCAN (sur PCA)", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# GMM
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)
plt.figure(figsize=(10, 8))
for cluster in range(optimal_k):
    cluster_data = X_pca[np.array(labels_gmm) == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                c=[colors[cluster]], label=f"GMM Cluster {cluster}", alpha=0.8, s=60)
plt.title("Clusters obtenus avec GMM (sur PCA)", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Stabilité K-Means
silhouettes_stability = []
for rs in range(10, 110, 10):
    kmeans_test = KMeans(n_clusters=optimal_k, random_state=rs, n_init=10)
    labels_test = kmeans_test.fit_predict(X_scaled)
    silhouettes_stability.append(silhouette_score(X_scaled, labels_test))

plt.figure(figsize=(10, 5))
plt.plot(range(10, 110, 10), silhouettes_stability, marker='o')
plt.title("Stabilité du modèle K-Means (Silhouette vs Random State)")
plt.xlabel("Random State")
plt.ylabel("Score de Silhouette")
plt.grid(True, alpha=0.3)
plt.show()

# Sauvegarde des résultats
df.to_csv('resultats_clustering.csv', index=False)
print(" Fichier créé : resultats_clustering.csv")
