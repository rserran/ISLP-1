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
from statsmodels.stats.outliers_influence import variance_inflation_factor

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# load data
url = "/Users/arpanganguli/Documents/Finance : ML/ISLR/Datasets/Boston.csv"
Boston = pd.read_csv(url, index_col = 'Unnamed: 0')
print(Boston.head())

# perform regression
Y = Boston['medv']
X1 = Boston['crim']
X2 = Boston['zn']
X3 = Boston['indus']
X4 = Boston['chas']
X5 = Boston['nox']
X6 = Boston['rm']
X7 = Boston['age']
X8 = Boston['dis']
X9 = Boston['rad']
X10 = Boston['tax']
X11 = Boston['ptratio']
X12 = Boston['black']
X13 = Boston['lstat']

model = ols("Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12", data = Boston).fit()
print(model.summary())

# calculate and display variance inflation factor
vif = pd.DataFrame()
vif["Variance Inflation Factor"] = [variance_inflation_factor(Boston.values, i)\
for i in range(Boston.shape[1])]
vif["Features"] = Boston.columns
print(vif["Variance Inflation Factor"])
print(vif["Features"])
