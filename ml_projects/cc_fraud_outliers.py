'''
Created on Mar 27, 2019

@author: dsj529

comparison of various outlier/novelty detection algorithms
data hosted at kaggle.com; the file must be downloaded for this code to work.
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

## load and inspect the data
data = pd.read_csv('../data/creditcard.csv')
# print(data.columns)
# data.hist(figsize=(20,20))
# plt.show()

## determins number of fraud cases in the data
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_frac = len(Fraud)/len(Valid)
print(outlier_frac)

print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Transactions: {}'.format(len(Valid)))

#===================================================================================================
# ## investigate correlation matrix
# corrmat = data.corr()
# fig = plt.figure(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.8, square=True)
# plt.show()
#===================================================================================================

## define the data for modeling
columns = data.columns.tolist()
columns = [c for c in columns if c not in ['Class']]
target = 'Class'
X = data[columns]
Y = data[target]
# print(X.shape)
# print(Y.shape)

classifiers = {
    'Isolation Forest': IsolationForest(n_estimators=250,
                                        max_samples=len(X),
                                        contamination=outlier_frac,
                                        behaviour='new'),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, 
                                               contamination=outlier_frac),
    'Robust Covariance': EllipticEnvelope(contamination=outlier_frac)}

## fit the models
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
#     y_pred = np.zeros(len(X))
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # reshape prediction values to 0/1 for valid/fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    
    # run metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))