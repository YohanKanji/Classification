#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 08:41:13 2020

@author: yohankanji
"""

import numpy as np, random, scipy.stats as ss

def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]

import pandas as pd

csv_data = pd.read_csv('/Users/dillipkanji/Desktop/2020/Desktop/Yohan/Python/wine.csv')

csv_data.rename(columns = {"color":"is_red"}, inplace = True)
csv_data = csv_data.replace(['red'],[1])
csv_data = csv_data.replace(['white'],[0])

red_count = 0
white_count = 0

for r in csv_data['is_red']:
    if r == 1:
        red_count += 1
        
for w in csv_data['is_red']:
    if w == 0:
        white_count += 1
        
import sklearn.preprocessing

csv_data.shape

numeric_data = csv_data
scaled_data = sklearn.preprocessing.scale(numeric_data)
numeric_data = pd.DataFrame(scaled_data, columns = numeric_data.columns)

import sklearn.decomposition

pca = sklearn.decomposition.PCA(n_components = 2)
principal_components = pca.fit_transform(numeric_data)
principal_components.shape
x = principal_components[:,0]
y = principal_components[:,1]

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages

observation_colormap = ListedColormap(['red', 'blue'])

plt.scatter(x, y, alpha = 0.2,
    c = numeric_data['high_quality'], cmap = observation_colormap, edgecolors = 'none')

