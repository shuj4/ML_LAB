import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]
# We only take the first two features for visualization
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit the algorithm to the data
kmeans.fit(X)
# Predict the cluster labels for the data points
labels = kmeans.labels_
# Get the coordinates of the cluster centers
centers = kmeans.cluster_centers_
# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-Means Clustering')
plt.show()