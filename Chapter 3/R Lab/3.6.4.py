#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:38:14 2018

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

# load data; visualisation same as Section 3.6.3
url = "/Users/arpanganguli/Documents/Finance : ML/ISLR/Datasets/Boston.csv"
Boston = pd.read_csv(url, index_col = 'Unnamed: 0')

# perform regression
Y = Boston['medv']
X1 = Boston['lstat']
X2 = Boston['age']
model = ols('Y~X1*X2', data = Boston).fit()
print(model.summary())

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESULTS (NOT PART OF CODE)

runfile('/Users/arpanganguli/Documents/Finance : ML/ISLR/Codes - Python/Section3.6.4.py', wdir='/Users/arpanganguli/Documents/Finance : ML/ISLR/Codes - Python')
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.556
Model:                            OLS   Adj. R-squared:                  0.553
Method:                 Least Squares   F-statistic:                     209.3
Date:                Thu, 27 Dec 2018   Prob (F-statistic):           4.86e-88
Time:                        18:10:36   Log-Likelihood:                -1635.0
No. Observations:                 506   AIC:                             3278.
Df Residuals:                     502   BIC:                             3295.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     36.0885      1.470     24.553      0.000      33.201      38.976
X1            -1.3921      0.167     -8.313      0.000      -1.721      -1.063
X2            -0.0007      0.020     -0.036      0.971      -0.040       0.038
X1:X2          0.0042      0.002      2.244      0.025       0.001       0.008
==============================================================================
Omnibus:                      135.601   Durbin-Watson:                   0.965
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              296.955
Skew:                           1.417   Prob(JB):                     3.29e-65
Kurtosis:                       5.461   Cond. No.                     6.88e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 6.88e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
