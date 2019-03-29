'''
Created on Mar 28, 2019

@author: dsj529

uses kmeans to classify digits in the MNIST dataset
'''
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans

import numpy as np


## load and inspect data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#===================================================================================================
# print('Training Data: {}'.format(X_train.shape))
# print('Training Labels: {}'.format(y_train.shape))
# print('Testing Data: {}'.format(X_test.shape))
# print('Testing Labels: {}'.format(y_test.shape))
# 
# ## look at sample images
# fig, axs = plt.subplots(3,3, figsize=(12,12))
# plt.gray()
#  
# for i, ax in enumerate(axs.flat):
#     ax.matshow(X_train[i])
#     ax.axis('off')
#     ax.set_title('Number {}'.format(y_train[i]))
# plt.show()
#===================================================================================================

## preprocess the images
X = X_train.reshape(len(X_train), -1)
y = y_train

X = X.astype(float) / 255.
# print(X.shape)
# print(X[0].shape)

n_digits = len(np.unique(y_test))
# print(n_digits)

kmeans = MiniBatchKMeans(n_clusters = n_digits)
kmeans.fit(X)

## associate clusters with labels
def infer_cluster_labels(model, actual_labels):
    '''
    Associates most probable label with each cluster in the model
    Returns: dict of clusters assigned to each label
    '''
    
    inferred_labels = {}
    for i in range(model.n_clusters):
        # find the index of points in cluster
        labels = []
        index = np.where(model.labels_ == i)
        
        # include actual labels for each point in cluster
        labels.append(actual_labels[index])
        
        # if multiple clusters, determine the most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
            
        #assign the cluster to a value in the inferred dict
        if np.argmax(counts) in inferred_labels:
            inferred_labels[np.argmax(counts)].append(i)
        else:
            inferred_labels[np.argmax(counts)] = [i]
            
    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
    '''
    Determines label for each array, based on the cluster it's been assigned to.
    Returns: predicted labels of each array
    '''
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for k, v in cluster_labels.items():
                if cluster in v:
                    predicted_labels[i] = k
    return predicted_labels

#===================================================================================================
# ## test the helper functions
# cluster_labels = infer_cluster_labels(kmeans, y)
# X_clusters = kmeans.predict(X)
# predicted_labels = infer_data_labels(X_clusters, cluster_labels)
# print(predicted_labels[:20])
# print(y[:20])
#===================================================================================================

## optimize

def calculate_metrics(estimator, labels):
    print('Number of clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}\n'.format(metrics.homogeneity_score(labels, estimator.labels_)))
    
clusters = [10, 16, 36, 64, 144, 256, 300]

for n in clusters:
    estimator = MiniBatchKMeans(n_clusters=n)
    estimator.fit(X)
    
    calculate_metrics(estimator, y)
    
    cluster_labels = infer_cluster_labels(estimator, y)
    predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)
    
    print('Accuracy: {}'.format(metrics.accuracy_score(y, predicted_Y)))
    
## test model
# reshape and normalize the test data
Xtest = X_test.reshape(len(X_test), -1)
Xtest = Xtest.astype(float) / 255.

kmeans = MiniBatchKMeans(n_clusters=256)
kmeans.fit(X)
cluster_labels = infer_cluster_labels(kmeans, y)

test_clusters = kmeans.predict(Xtest)
predicted_labels = infer_data_labels(kmeans.predict(Xtest), cluster_labels)
print('Test Accuracy: {}\n'.format(metrics.accuracy_score(y_test, predicted_labels)))

## visualize the centroids
kmeans = MiniBatchKMeans(n_clusters=36)
kmeans.fit(X)

centroids = kmeans.cluster_centers_

# reshape the centroids into images
images = centroids.reshape(36,28,28)
images *= 255
images = images.astype(np.uint8)

# determine labels
cluster_labels = infer_cluster_labels(kmeans, y)

fig, axs = plt.subplots(6,6, figsize=(20,20))
plt.gray()

for i, ax in enumerate(axs.flat):
    for k, v in cluster_labels.items():
        if i in v:
            ax.set_title('Inferred Label: {}'.format(k))
    ax.matshow(images[i])
    ax.axis('off')
    
plt.show()