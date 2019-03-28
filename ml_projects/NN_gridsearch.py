'''
Created on Feb 12, 2019

@author: dajoseph
'''
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from tensorflow.python.training import learning_rate_decay
from nltk.sem.logic import equality_preds

## collect data from server
url = ('http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
# print(url)
names = ['wf_make', 'wf_address', 'wf_all', 'wf_3d', 'wf_our', 'wf_over', 'wf_remove',
         'wf_internet', 'wf_order', 'wf_mail', 'wf_receive', 'wf_will', 'wf_people', 'wf_report',
         'wf_addresses', 'wf_free', 'wf_business', 'wf_email', 'wf_you', 'wf_credit', 'wf_your',
         'wf_font', 'wf_000', 'wf_money', 'wf_hp', 'wf_hpl', 'wf_george', 'wf_650', 'wf_lab', 
         'wf_labs', 'wf_telnet', 'wf_857', 'wf_data', 'wf_415', 'wf_85', 'wf_technology', 'wf_1999',
         'wf_parts', 'wf_pm', 'wf_direct', 'wf_cs', 'wf_meeting', 'wf_original', 'wf_project',
         'wf_re', 'wf_edu', 'wf_table', 'wf_conference', 'cf_;', 'cf_(', 'cf_[', 'cf_!', 'cf_$',
         'cf_#', 'consec_caps_avg', 'consec_caps_longest', 'consec_caps_total', 'class']
df = pd.read_csv(url.strip(), names=names)

## convert to numpy arrays for processing
dataset = df.values
X = dataset[:,:-1]
Y = dataset[:, -1].astype(int)
print('full: {}\tX: {}\tY: {}'.format(dataset.shape, X.shape, Y.shape))

## Normalize the data
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)
data = pd.DataFrame(X_std)
# print(data.describe())

## set up the model
def create_model(neurons, learn_rate, drop_rate):
    model = Sequential()
    model.add(Dense(57, input_dim=57, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


## set up model test conditions
model = KerasClassifier(build_fn=create_model, epochs=25, verbose=0)

param_grid = {'batch_size': [50, 100, 200, 500],
              'learn_rate': [0.0, 0.1, 0.2],
              'drop_rate': [0.0, 0.05, 0.1, 0.5],
              'neurons': [8, 10, 15, 20, 50]}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=10)
grid_results = grid.fit(X_std, Y)

print("Best: {0} using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
    
y_pred = grid.predict(X_std)
print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))
