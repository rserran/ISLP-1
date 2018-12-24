#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:15:26 2018

@author: arpanganguli
"""

# import statistical tools
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# load data
url = "/Users/arpanganguli/Documents/Finance : ML/ISLR/Datasets/Boston.csv"
Boston = pd.read_csv(url, index_col = 'Unnamed: 0')
print(Boston.head())

# perform regression
Y = Boston['medv']
X1 = Boston['lstat']
X2 = Boston['age']
model = ols("Y ~ X1 + X2", data = Boston).fit()
print(model.summary())
