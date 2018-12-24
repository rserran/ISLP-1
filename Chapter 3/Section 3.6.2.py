# 3.6.2 Simple Linear Regression
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 02:29:16 2018

@author: arpanganguli
"""

# import statistical tools
import numpy as np
import pandas as pd
import sklearn
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset.
Boston = pd.read_csv("/Users/arpanganguli/Documents/Finance : ML/ISLR/Datasets/Boston.csv", index_col = "Unnamed: 0")
"""
use the absolute path in your own computer like I have used in mine above
"""

# basic exploration of data
print(Boston.head())
print(Boston.corr())

# fit model through linear regression
Y = Boston['medv']
X = Boston['lstat']
model = ols("Y~X", data = Boston).fit()
print(model.summary())

# predict the model
dt = summary_table(model, alpha = 0.5)[1]
Y_prd = dt[:, 2]
Yprd_ci_lower, Yprd_ci_upper = dt[:, 6:8].T
print(pd.DataFrame(np.column_stack([Y_prd, Yprd_ci_lower, Yprd_ci_upper])).head())

# plot graph with regression line
plt.figure(2).add_subplot(121)
print(sns.regplot(X, Y, data = model, color = 'g'))
plt.figure(2).add_subplot(122)
print(sns.residplot(X, Y, lowess = True, color = 'r'))
