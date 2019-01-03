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
url = "/Users/arpanganguli/Documents/Finance/ISLR/Datasets/Carseats.csv"
CarSeats = pd.read_csv(url)
print(CarSeats.head())
print(list(CarSeats))
print(CarSeats.info())

# run regression
reg = ols(formula = 'Sales ~ Price + C(Urban) + C(US)', data = CarSeats).fit()
print(reg.summary())
