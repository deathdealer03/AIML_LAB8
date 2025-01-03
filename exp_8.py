import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset from CSV file
# Replace 'health_data.csv' with the path to your CSV file
csv_file = 'health_data.csv'
df = pd.read_csv(csv_file)

# Selecting any 4 numeric columns
# In this example: ['Age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level']
selected_columns = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level']
X = df[selected_columns].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

# Print the first few rows of the transformed DataFrame
print("First few rows after PCA transformation:")
print(df_pca.head())

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(f"\nExplained variance by each principal component: {explained_variance_ratio}")
print(f"\nTotal variance explained by the two components: {sum(explained_variance_ratio)}")

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c='b', s=50, alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Selected Numeric Columns')
plt.grid(True)
plt.show()

# Full PCA for cumulative variance
pca_full = PCA(n_components=X.shape[1])
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative explained variance for all principal components:")
print(cumulative_variance)

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, X.shape[1] + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance as a Function of the Number of Components')
plt.grid(True)
plt.show()
