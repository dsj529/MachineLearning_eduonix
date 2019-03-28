'''
Created on Mar 27, 2019

@author: dsj529
'''
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

## Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)

## Preprocess the data
df.replace('?',-99999, inplace=True)
print(df.axes)

df.drop(['id'], 1, inplace=True)

#===================================================================================================
# ## inspect the dataset
# print(df.loc[10])
# print(df.shape)
# print(df.describe())
# df.hist(figsize=(10,10))
# plt.show()
# scatter_matrix(df, figsize=(18,18))
# plt.show()
#===================================================================================================

## prepare data for learning
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


## set up the classifiers
scoring = 'accuracy'

models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC(gamma='scale')))

results=[]
names=[]

for name, model in models:
    kfold=model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{}: {:f} ({:f})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)
    
## predict on test/validation set
for name, model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))