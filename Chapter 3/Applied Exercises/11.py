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

# 11.f. run regression WITH intercept
reg_3 = ols("y~x", data = data).fit()
print(reg_3.summary())

reg_4 = ols("x~y", data = data).fit()
print(reg_4.summary())
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESULTS (NOT PART OF CODE)

runfile('/Users/arpanganguli/Documents/Finance/ISLR/General_Code.py', wdir='/Users/arpanganguli/Documents/Finance/ISLR')
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.770
Model:                            OLS   Adj. R-squared:                  0.767
Method:                 Least Squares   F-statistic:                     330.9
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           2.46e-33
Time:                        03:01:25   Log-Likelihood:                -144.94
No. Observations:                 100   AIC:                             291.9
Df Residuals:                      99   BIC:                             294.5
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x              2.0371      0.112     18.191      0.000       1.815       2.259
==============================================================================
Omnibus:                        4.100   Durbin-Watson:                   2.108
Prob(Omnibus):                  0.129   Jarque-Bera (JB):                3.424
Skew:                          -0.382   Prob(JB):                        0.180
Kurtosis:                       3.488   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      x   R-squared:                       0.770
Model:                            OLS   Adj. R-squared:                  0.767
Method:                 Least Squares   F-statistic:                     330.9
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           2.46e-33
Time:                        03:01:25   Log-Likelihood:                -60.705
No. Observations:                 100   AIC:                             123.4
Df Residuals:                      99   BIC:                             126.0
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
y              0.3779      0.021     18.191      0.000       0.337       0.419
==============================================================================
Omnibus:                        4.479   Durbin-Watson:                   2.112
Prob(Omnibus):                  0.107   Jarque-Bera (JB):                3.796
Skew:                           0.428   Prob(JB):                        0.150
Kurtosis:                       3.422   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.769
Model:                            OLS   Adj. R-squared:                  0.767
Method:                 Least Squares   F-statistic:                     326.0
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           6.13e-33
Time:                        03:01:25   Log-Likelihood:                -144.25
No. Observations:                 100   AIC:                             292.5
Df Residuals:                      98   BIC:                             297.7
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.1214      0.104     -1.170      0.245      -0.327       0.085
x              2.0258      0.112     18.057      0.000       1.803       2.248
==============================================================================
Omnibus:                        3.991   Durbin-Watson:                   2.137
Prob(Omnibus):                  0.136   Jarque-Bera (JB):                3.315
Skew:                          -0.375   Prob(JB):                        0.191
Kurtosis:                       3.482   Cond. No.                         1.12
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      x   R-squared:                       0.769
Model:                            OLS   Adj. R-squared:                  0.767
Method:                 Least Squares   F-statistic:                     326.0
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           6.13e-33
Time:                        03:01:25   Log-Likelihood:                -60.513
No. Observations:                 100   AIC:                             125.0
Df Residuals:                      98   BIC:                             130.2
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.0277      0.045      0.614      0.541      -0.062       0.117
y              0.3796      0.021     18.057      0.000       0.338       0.421
==============================================================================
Omnibus:                        4.565   Durbin-Watson:                   2.120
Prob(Omnibus):                  0.102   Jarque-Bera (JB):                3.883
Skew:                           0.432   Prob(JB):                        0.144
Kurtosis:                       3.429   Cond. No.                         2.18
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
