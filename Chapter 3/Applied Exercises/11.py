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

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# set seed
random.seed(1)
x= pd.DataFrame(np.random.normal(0, 1, 100))
y = 2*x + pd.DataFrame(np.random.normal(0, 1, 100))

data = pd.concat([x,y], axis = 1)

# 11.a. run regression y~x
reg_1 = ols("y~x+0", data = data).fit()
print(reg_1.summary())

# 11.b. run regression x~y
reg_2 = ols("x~y+0", data = data).fit()
print(reg_2.summary())
