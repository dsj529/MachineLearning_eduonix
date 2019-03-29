'''
Created on Mar 28, 2019

@author: dsj529

Simple classifier of SMS text messages into spam/not-spam classes.
Data is downloaded from https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

By using a POS-informed lemmatization of words instead of the Porter stemmer
as done in the original tutorial, I have seen the overall accuracy of this project rise
by ~2.5 percentage points.
'''
from collections import defaultdict

from nltk import word_tokenize, pos_tag
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd


TAG_MAP = defaultdict(lambda: wn.NOUN)
TAG_MAP['J'] = wn.ADJ
TAG_MAP['V'] = wn.VERB
TAG_MAP['R'] = wn.ADV
TAG_MAP['S'] = wn.ADJ_SAT

WNL = WordNetLemmatizer()

def lemmatize_msg(msg):
    tokenized = word_tokenize(msg)
    lemmatized = [WNL.lemmatize(token, TAG_MAP[tag[0]])
                  for token, tag in pos_tag(tokenized)]
    return ' '.join(lemmatized)                            

df = pd.read_csv('../data/SMSSpamCollection', header=None, encoding='utf-8', sep='\t')
# print(df.info())
# print(df.head())

classes = df[0]
# print(classes.value_counts())

## turn categorical labels into numeric ones for easier computation
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

## start preprocessing the messages
text_messages = df[1]

# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                      'emailaddr')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddr')

# Replace money symbols with 'moneysymb' 
processed = processed.str.replace(r'£|\$|¥|€', 'moneysymb')

# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'usphone'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'usphone')

# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Strip punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Condense whitespace between terms
processed = processed.str.replace(r'\s+', ' ')

# Strip leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

# cast down to lowercase
processed = processed.str.lower()

# remove stopwords
stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() 
                                               if term not in stop_words))

processed = processed.apply(lemmatize_msg)

# print(processed[:15])

## create bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)
print('Total Words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15))) 

word_features = list(all_words.keys())[:1500]

## define a function to compare message text to the features list
def find_features(msg):
    words = word_tokenize(msg)
    features={}
    for word in word_features:
        features[word] = (word in words)
    return features

#===================================================================================================
# features = find_features(processed[0])
# for k, v in features.items():
#     if v: print(k)
#===================================================================================================

## turn the processed data into test and train sets
messages = list(zip(processed, Y))

np.random.shuffle(messages)

feature_sets = [(find_features(msg), label) for (msg, label) in messages]

training, testing = model_selection.train_test_split(feature_sets, test_size=0.25)

print(len(training))
print(len(testing))

## start learning the data with various models
model = SklearnClassifier(SVC(kernel='linear'))
model.train(training)
accuracy = nltk.classify.accuracy(model, testing)
print('SVC Accuracy: {:.4%}'.format(accuracy))


names = ['K-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression',
         'SGD [Linear] Classifier', 'Naive Bayes', 'SVM Linear']
classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=250),
               LogisticRegression(solver='lbfgs'),
               SGDClassifier(max_iter=1000, tol=1e-3),
               MultinomialNB(),
               SVC(kernel='linear')]
models = list(zip(names, classifiers))

print('Evaluating individual models')
for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)
    print('\t{} Accuracy: {:.4%}'.format(name, accuracy))
    
print('Evaluating hard-voting ensemble of models')
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models, voting='hard', n_jobs=-1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)
print('\tEnsemble Voting Classifier Accuracy: {:.4%}'.format(accuracy))

## test making predictions on the data
txt_features, labels = zip(*testing)
preds = nltk_ensemble.classify_many(txt_features)

print(classification_report(labels, preds))
print(pd.DataFrame(confusion_matrix(labels, preds),
                   index=[['actual', 'actual'], ['ham', 'spam']],
                   columns = [['predicted', 'predicted'], ['ham', 'spam']]))
