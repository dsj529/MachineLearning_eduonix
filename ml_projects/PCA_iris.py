'''
Created on Mar 28, 2019

@author: dsj529

clustering of the Iris dataset using kmeans and PCA
'''

from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd


iris = datasets.load_iris()
features = iris.data
target = iris.target

df = pd.DataFrame(features)
df.columns = iris.feature_names

# print(df.describe())
# scatter_matrix(df)
# plt.show()

#===================================================================================================
# ## Use the Elbow method to find optimal cluster number
# X = []
# Y = []
# 
# for i in range(1,31):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(df)
#     
#     # append the number of clusters to X, and average within-cluster sum of squares to Y
#     X.append(i)
#     awcss = kmeans.inertia_ / df.shape[0]
#     Y.append(awcss)
#     
# plt.plot(X, Y, 'bo-')
# plt.xlim((1,30))
# plt.xlabel('Number of Clusters')
# plt.ylabel('Average Within-Cluster Sum of Squares')
# plt.title('KMeans Clustering Elbow method')
# plt.show()
#===================================================================================================


pca = PCA(n_components=2)
pc = pca.fit_transform(df)
# print(pc.shape)
# print(pc[:10])

# fit with 3 clusters as per elbow graph
kmeans = KMeans(n_clusters=3)
kmeans.fit(pc)

## visualize the clusters
# create an XY mesh
h = 0.02
x_min, x_max = pc[:, 0].min() - 1, pc[:, 0].max() + 1
y_min, y_max = pc[:, 1].min() - 1, pc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# plot the results on the mesh grid
Z = Z.reshape(xx.shape)
plt.figure(figsize=(12,12))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.tab20c,
           aspect='auto', origin='lower')

# plot the principal components
for i, point in enumerate(pc):
    if target[i] == 0:
        plt.plot(point[0], point[1], 'g.', markersize=10)
    if target[i] == 1:
        plt.plot(point[0], point[1], 'r.', markersize=10)
    if target[i] == 2:
        plt.plot(point[0], point[1], 'b.', markersize=10)

# plot the centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=250, linewidth=4,
            color='w', zorder=10)

# label the plot
plt.title('KMeans Clustering on PCA-Reduced Iris Data Set')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xticks(())
plt.yticks(())
plt.show()

## Evaluate with clustering metrics
## Homogeneity: how consistently each cluster contains only one label
## Completeness: how consistently each label belongs to only one cluster
## V-measure: harmonic mean of Homogeneity and Completeness
kmeans1 = KMeans(n_clusters=3)
kmeans1.fit(features)

kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(pc)

print('Original data')
print('\tHomogeneity: {}'.format(metrics.homogeneity_score(target, kmeans1.labels_)))
print('\tCompleteness: {}'.format(metrics.completeness_score(target, kmeans1.labels_)))
print('\tV-measure: {}'.format(metrics.v_measure_score(target, kmeans1.labels_)))

print('PCA-Reduced data')
print('\tHomogeneity: {}'.format(metrics.homogeneity_score(target, kmeans2.labels_)))
print('\tCompleteness: {}'.format(metrics.completeness_score(target, kmeans2.labels_)))
print('\tV-measure: {}'.format(metrics.v_measure_score(target, kmeans2.labels_)))