#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:41:50 2018
@author: arpanganguli
"""

# import statistical tools
from __future__ import print_function
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# import data visualisation tools
import matplotlib.pyplot as plt
from matplotlib import pylab
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns

# import and view first 10 items of file
url = "/Users/arpanganguli/Documents/Finance/ISLR/Datasets/Auto.csv"
Auto = pd.read_csv(url)
print(Auto.head())
print(list(Auto))
Auto = Auto.drop(Auto.index[[32, 126, 330, 336, 354]])
"""
removing rows containing "?". This is the easy way out. Such missing
values need to be explored first in a real life situation.
"""
print(Auto.info())

# 9.a. Scatterplot Matrix
sns.pairplot(Auto, hue = "origin")

# 9.b. Correlation Matrix
Correlation_Matrix = Auto.corr()
print(Correlation_Matrix)
