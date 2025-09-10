#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
This script performs clustering analysis on fish morphological data using
K-Means, DBSCAN, and Hierarchical clustering algorithms. It includes
visualization of results and evaluation metrics.

Requirements:
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- plotly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# Load and inspect the data
data = pd.read_csv("fish_data.csv")
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Display dataset information
print("\nDataset info:")
data.info()

# Remove species column for unsupervised learning
data = data.drop(["species"], axis=1)
print("\nData after removing species column:")
print(data.head())

# Elbow method and silhouette score analysis for K-Means
X = data
sse = []  # Sum of squared errors
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot elbow method results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.scatter(range(1, 11), sse)
plt.grid()
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal K')
plt.show()

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.scatter(range(2, 11), silhouette_scores)
plt.grid()
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Numbers')
plt.show()

# Apply K-Means clustering with optimal number of clusters (3)
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# Visualize K-Means results in 3D
fig = px.scatter_3d(X, x='length', y='weight', z='w_l_ratio',
                    color=labels, title='K-Means Clustering Results')
fig.show()

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=15, min_samples=10)
dbscan.fit(X)
labels = dbscan.labels_

# Visualize DBSCAN results in 3D
fig = px.scatter_3d(X, x=X['length'], y=X['weight'], z=X['w_l_ratio'], 
                    color=labels, title='DBSCAN Clustering Results')
fig.show()

# Perform hierarchical clustering
linked = linkage(X, 'ward')
plt.figure(figsize=(15, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

