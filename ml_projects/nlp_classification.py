'''
Created on Mar 28, 2019

@author: dajoseph

a simple text classifier built over NLTK
'''

import random

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from sklearn import model_selection
from sklearn.svm import SVC


docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)

# print('Number of Documents: {}'.format(len(docs)))
# print('First Review: {}'.format(documents[1]))

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words) # could import Counter from collections and implement directly
# print('most common words: {}'.format(all_words.most_common(15)))
# print('the word "happy" is the {} most popular word'.format(allwords['happy']))
# print(len(all_words))

word_features = list(all_words.keys())[:4000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#===================================================================================================
# features = find_features(movie_reviews.words('neg/cv000_29416.txt'))
# for k, v in features.items():
#     if v == True: print(k)
#===================================================================================================

## build the test/train data sets, define the model and go
featuresets = [(find_features(rev), category) for (rev, category) in docs]

training, testing = model_selection.train_test_split(featuresets, test_size=0.25)

# print(len(training))
# print(len(testing))


model = SklearnClassifier(SVC(kernel='linear'))
model.train(training)
accuracy = nltk.classify.accuracy(model, testing)
print('SVC Accuracy: {:.4%}'.format(accuracy))