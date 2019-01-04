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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESULTS (NOT PART OF CODE)

runfile('/Users/arpanganguli/Documents/Finance/ISLR/General_Code.py', wdir='/Users/arpanganguli/Documents/Finance/ISLR')
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.806
Model:                            OLS   Adj. R-squared:                  0.804
Method:                 Least Squares   F-statistic:                     412.2
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           4.55e-37
Time:                        02:57:15   Log-Likelihood:                -138.29
No. Observations:                 100   AIC:                             278.6
Df Residuals:                      99   BIC:                             281.2
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x              1.9142      0.094     20.303      0.000       1.727       2.101
==============================================================================
Omnibus:                        3.395   Durbin-Watson:                   2.044
Prob(Omnibus):                  0.183   Jarque-Bera (JB):                3.218
Skew:                           0.186   Prob(JB):                        0.200
Kurtosis:                       3.796   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      x   R-squared:                       0.806
Model:                            OLS   Adj. R-squared:                  0.804
Method:                 Least Squares   F-statistic:                     412.2
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           4.55e-37
Time:                        02:57:15   Log-Likelihood:                -62.596
No. Observations:                 100   AIC:                             127.2
Df Residuals:                      99   BIC:                             129.8
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
y              0.4212      0.021     20.303      0.000       0.380       0.462
==============================================================================
Omnibus:                        0.204   Durbin-Watson:                   2.146
Prob(Omnibus):                  0.903   Jarque-Bera (JB):                0.134
Skew:                           0.088   Prob(JB):                        0.935
Kurtosis:                       2.965   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
