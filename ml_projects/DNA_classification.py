'''
Created on Mar 4, 2019

implements multiple classification algorithms within the context of DNA sequencing

@author: dajoseph
'''
import re

import numpy as np
import scipy as sp
import pandas as pd

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# import data from UCI ML repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
names = ['Class', 'id', 'Sequence']
data = pd.read_csv(url, names = names)

# transform the data into a useful dataframe
classes = data['Class']
sequences = list(data['Sequence'])
dataset = {}
for i, seq in enumerate(sequences):
    # split into nucleotides and remove tabs
    nucleotides = list(re.sub(r'\t', r'', seq))
    
    # append class assignment
    nucleotides.append(classes[i])
    
    # add to dataset
    dataset[i] = nucleotides

df = pd.DataFrame(dataset).T
df.rename(columns={57: "Class"}, inplace=True)
# print(dframe.head())
# print(df.describe())

#===================================================================================================
# # look at frequencies of each nucleotide per sequence
# series = []
# for name in df.columns:
#     series.append(df[name].value_counts())
# info = pd.DataFrame(series).T
# print(info)
#===================================================================================================

df = pd.get_dummies(df)
print(df.columns)
df = df.drop(columns=['Class_-'])
df.rename(columns={'Class_+': 'Class'}, inplace=True)
print(df.head())

# Test and train the models
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

scoring = 'accuracy'

names = ['Nearest Neighbors', 'Gaussian Process', 'Decision Tree', 'Random Forest',
         'Neural Net', 'AdaBoost', 'Naive Bayes', 'SVM Linear', 'SVM RBF', 'SVM Sigmoid']

classifiers = [KNeighborsClassifier(n_neighbors=3),
               GaussianProcessClassifier(1.0 * RBF(1.0)),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=25, max_features=1),
               MLPClassifier(alpha=1),
               AdaBoostClassifier(),
               GaussianNB(),
               SVC(kernel='linear', gamma='scale'),
               SVC(kernel='rbf', gamma='scale'),
               SVC(kernel='sigmoid', gamma='scale')]

models = zip(names, classifiers)
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{}: {:0.4} ({:0.4})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)