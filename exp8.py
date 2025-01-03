import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = iris.target_names[y]

print("First few rows of the dataset:")
print(df.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = iris.target_names[y]

print("\nFirst few rows after PCA transformation:")
print(df_pca.head())

explained_variance_ratio = pca.explained_variance_ratio_
print(f"\nExplained variance by each principal component: {explained_variance_ratio}")
print(f"\nTotal variance explained by the two components: {sum(explained_variance_ratio)}")

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
species = ['setosa', 'versicolor', 'virginica']

for i, species_name in enumerate(species):
    subset = df_pca[df_pca['species'] == species_name]
    plt.scatter(subset['PC1'], subset['PC2'], label = species_name, c=colors[i], s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of the Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()

pca_full = PCA(n_components = X.shape[1])
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative explained variance for all principal components:")
print(cumulative_variance)

plt.figure(figsize=(8, 6))
plt.plot(range(1, X.shape[1] + 1), cumulative_variance, marker = 'o', linestyle='--', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance as a Function of the Number of Components')
plt.grid(True)
plt.show()