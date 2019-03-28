'''
Created on Dec 11, 2018

@author: dajoseph

NOTE: this project was originally written to use pandas_datareader and yahoo finance.
Yahoo has significantly altered its data api, and pandas_datareader has deprecated many
of its readers, so I have elected to rewrite the data acquisition portion of this project
using the Quandl api instead.
'''
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import quandl
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

quandl.read_key('../data/quandl.key') # personal key to quandl API

company_dict = {
    'Amazon': 'AMZN',
    'Apple': 'AAPL',
    'Walgreen': 'WBA',
    'Northrop Grumman': 'NOC',
    'Boeing': 'BA',
    'Lockheed Martin': 'LMT',
    'McDonalds': 'MCD',
    'Intel': 'INTC',
    'Navistar': 'NAV',  
    'IBM': 'IBM',
    'Texas Instruments': 'TXN',
    'MasterCard': 'MA',
    'Microsoft': 'MSFT',
    'General Electrics': 'GE',
    'Symantec': 'SYMC',
    'American Express': 'AXP',
    'Pepsi': 'PEP',
    'Coca Cola': 'KO',
    'Johnson & Johnson': 'JNJ',
    'Toyota': 'TM',
    'Honda': 'HMC',
    'Mitsubishi': 'MSBHY',
    'Sony': 'SNE',
    'Exxon': 'XOM',
    'Chevron': 'CVX',
    'Valero Energy': 'VLO',
    'Ford': 'F',
    'Bank of America': 'BAC'}

all_companies = sorted(company_dict.items(), key=lambda x: x[1]) # sort list based on stock ticker symbol.
start_date ='2010-07-01'
end_date = '2018-06-30'

raw_data = quandl.get_table('WIKI/PRICES', ticker=list(company_dict.values()),
                        qopts={'columns': ['ticker', 'date', 'open', 'close']},
                        date={'gte': '2014-07-01', 'lte': '2018-06-30'},
                        paginate=True)

raw_data['movement'] = raw_data.close - raw_data.open
use_comps = [c for c in all_companies if c[1] in raw_data.ticker.values]

stock_movements = (raw_data[['ticker', 'date', 'movement']]
                  .pivot_table(values='movement', index='date', columns='ticker')
                  .fillna(0)
                  .T)

## first look at how the full data clusters together
normalizer = Normalizer()
kmeans = KMeans(n_clusters=10, max_iter=2500)
pipeline = make_pipeline(normalizer, kmeans)

pipeline.fit(stock_movements)
print(kmeans.inertia_)

labels = pipeline.predict(stock_movements)
df1 = pd.DataFrame({'labels': labels, 'companies': use_comps})
print(df1.sort_values('labels'))

## next, look at lower-dimensioned data via PCA reduction
normed_movements = normalizer.fit_transform(stock_movements)
reduced_data = PCA(n_components=2).fit_transform(normed_movements)
kmeans2 = KMeans(init='k-means++', n_clusters=10, n_init=10, max_iter=2500)
kmeans2.fit(reduced_data)
labels2 = kmeans2.predict(reduced_data)

df2 = pd.DataFrame({'labels': labels, 'companies': use_comps})
print(df2.sort_values('labels'))

## create a visualization of the reduced-dimensionality data
h = 0.001

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:,0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans2.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape) # put the results into a plot

cmap = plt.cm.Paired

plt.figure(figsize=(10,10))
plt.clf()
plt.imshow(Z, interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=cmap,
           aspect='auto', origin='lower')
plt.plot(reduced_data[:,0], reduced_data[:, 1], 'k.', markersize=5)
centroids = kmeans2.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('K-Means clustering on stock market movement data (reduced to 2 PCA dimensions)\n'
          'Centroids are marked with white cross')
plt.show()
