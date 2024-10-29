import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate sample data
X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

# Initialize DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)

# Fit the model and predict clusters
clusters = dbscan.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

from sklearn.neighbors import NearestNeighbors

# Calculate distances to the nearest 4 neighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Sort and plot distances
distances = np.sort(distances[:, 4], axis=0)
plt.plot(distances)
plt.xlabel("Data Points Sorted by Distance")
plt.ylabel("4th Nearest Neighbor Distance")
plt.title("k-distance Graph for DBSCAN")
plt.show()
