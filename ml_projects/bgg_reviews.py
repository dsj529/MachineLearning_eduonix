'''
Created on Mar 27, 2019

@author: dsj529

uses data from https://github.com/ThaWeatherman/scrapers.git
'''
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

## load and inspect the data
games = pd.read_csv('../data/bgg_games_data.csv')
# print(games.columns)
# print(games.shape)
# plt.hist(games['average_rating'])
# plt.show()

# remove games without reviews
games = games[games['users_rated'] > 0]
# remove games with missing data
games = games.dropna(axis=0)

# plt.hist(games['average_rating'])
# plt.show()

#===================================================================================================
# ## look deeper into the data for correlations between columns
# corrmat = games.corr()
# fig = plt.figure(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.8, square=True)
# plt.show()
#===================================================================================================

## filter correlated columns
cols = games.columns.tolist()
cols = [c for c in cols if c not in ['bayes_average_rating', 'average_rating', 'type', 'name', 'id']]

target = 'average_rating'

## build test/train split of the data
train = games.sample(frac=0.8)
test = games.loc[~games.index.isin(train.index)]
# print(train.shape)
# print(test.shape)

model = LinearRegression()
model.fit(train[cols], train[target])
preds = model.predict(test[cols])
print('Linear Regression: ', mean_squared_error(preds, test[target]))

model = RandomForestRegressor(n_estimators=250, min_samples_leaf=5)
model.fit(train[cols], train[target])
preds = model.predict(test[cols])
print('Random Forest: ', mean_squared_error(preds, test[target]))