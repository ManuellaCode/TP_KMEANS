print("\n\nPARTIE 2 : ANALYSE EXPLORATOIRE")
print("=" * 50)
# 2.1 Analyse de corrélation entre les variables numériques
print("\n2.1 HEATMAP DE CORRÉLATION")

plt.figure(figsize=(10, 8))
mat_corr = df[features].corr()

sns.heatmap(
    mat_corr,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    fmt='.2f',
    cbar_kws={"shrink": 0.8}
)

plt.title("Matrice de Corrélation des Variables", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("Matrice de corrélation :")
print(mat_corr)
# 2.2 Visualisation par diagrammes de dispersion (scatter plots)
print("\n2.2 SCATTER PLOTS")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Âge vs Score de dépense (coloré selon le revenu)
plot_age_spend = axes[0].scatter(
    df['Age'],
    df['Spending_Score_(1-100)'],
    c=df['Annual_Income_(k$)'],
    cmap='viridis',
    alpha=0.7,
    s=60
)
axes[0].set_xlabel("Âge")
axes[0].set_ylabel("Score de Dépense")
axes[0].set_title("Âge vs Score de Dépense\n(Couleur = Revenu)", fontweight='bold')
plt.colorbar(plot_age_spend, ax=axes[0])

# Revenu vs Score de dépense (coloré selon l’âge)
plot_income_spend = axes[1].scatter(
    df['Annual_Income_(k$)'],
    df['Spending_Score_(1-100)'],
    c=df['Age'],
    cmap='plasma',
    alpha=0.7,
    s=60
)
axes[1].set_xlabel("Revenu Annuel (k$)")
axes[1].set_ylabel("Score de Dépense")
axes[1].set_title("Revenu vs Score de Dépense\n(Couleur = Âge)", fontweight='bold')
plt.colorbar(plot_income_spend, ax=axes[1])

# Âge vs Revenu (coloré selon le score de dépense)
plot_age_income = axes[2].scatter(
    df['Age'],
    df['Annual_Income_(k$)'],
    c=df['Spending_Score_(1-100)'],
    cmap='cool',
    alpha=0.7,
    s=60
)
axes[2].set_xlabel("Âge")
axes[2].set_ylabel("Revenu Annuel (k$)")
axes[2].set_title("Âge vs Revenu\n(Couleur = Score de Dépense)", fontweight='bold')
plt.colorbar(plot_age_income, ax=axes[2])

plt.tight_layout()
plt.show()

# 2.3 Pairplot pour une vue d’ensemble des relations
print("\n2.3 PAIRPLOT")

pairplot_fig = sns.pairplot(
    df[features],
    diag_kind='hist',
    plot_kws={'alpha': 0.6, 's': 50}
)
pairplot_fig.fig.suptitle("Pairplot des Variables Numériques", y=1.02, fontsize=16, fontweight='bold')
plt.show()

# 2.4 Interprétation visuelle et hypothèses de regroupement
print("\n2.4 HYPOTHÈSES SUR LES CLUSTERS")
print("D'après les visualisations, on peut remarquer :")
print("• L’existence de plusieurs groupes bien distincts dans le graphique Revenu vs Score de Dépense.")
print("• Cela suggère probablement quatre clusters naturels :")
print("  1. Jeunes à faible revenu mais dépensiers.")
print("  2. Jeunes à revenu élevé avec des dépenses modérées.")
print("  3. Seniors à revenu faible et dépenses limitées.")
print("  4. Seniors à revenu élevé et dépenses modérées à élevées.")
