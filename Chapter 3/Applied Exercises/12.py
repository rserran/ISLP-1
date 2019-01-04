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

# 12.a. sum(y^2) = sum(x^2)

#12.b. regressions with different coefficients

x1 = pd.DataFrame(np.random.normal(0, 1, 100))
y1 = 2*x1

# regression 1
data_1= pd.concat([x1, y1], axis = 1)
reg_x = ols("y1~x1+0", data = data_1).fit()
print(reg_x.summary())

# regression 2
reg_y = ols("x1~y1+0", data = data_1).fit()
print(reg_y.summary())
print()

"""
Same coefficients
"""

# 12.c. regression with same coefficients

x2 = pd.DataFrame(np.random.normal(0, 1, 100))
y2 = pd.DataFrame(np.random.choice(x2[0], size = 100, replace = False))

print("Summation x^2: %f" % np.sum(pow(x2[0], 2)))
print("Summation y^2: %f\n" % np.sum(pow(y2[0], 2)))

# regression 1
data_2 = pd.concat([x2, y2], axis = 1)
reg_x = ols("y2~x2+0", data = data_2).fit()
print(reg_x.summary())
plt.figure(figsize = (11, 5))

# regression 2
reg_y = ols("x2~y2+0", data = data_2).fit()
print(reg_y.summary())

"""
Different coefficients
"""
