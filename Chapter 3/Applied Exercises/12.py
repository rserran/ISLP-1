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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESULTS (NOT PART OF CODE)

runfile('/Users/arpanganguli/Documents/Finance/ISLR/General_Code.py', wdir='/Users/arpanganguli/Documents/Finance/ISLR')
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                     y1   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                       inf
Date:                Fri, 04 Jan 2019   Prob (F-statistic):               0.00
Time:                        16:19:39   Log-Likelihood:                    inf
No. Observations:                 100   AIC:                              -inf
Df Residuals:                      99   BIC:                              -inf
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             2.0000          0        inf      0.000       2.000       2.000
==============================================================================
Omnibus:                        1.044   Durbin-Watson:                     nan
Prob(Omnibus):                  0.593   Jarque-Bera (JB):               37.500
Skew:                           0.000   Prob(JB):                     7.19e-09
Kurtosis:                       0.000   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                     x1   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                       inf
Date:                Fri, 04 Jan 2019   Prob (F-statistic):               0.00
Time:                        16:19:39   Log-Likelihood:                    inf
No. Observations:                 100   AIC:                              -inf
Df Residuals:                      99   BIC:                              -inf
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
y1             0.5000          0        inf      0.000       0.500       0.500
==============================================================================
Omnibus:                        1.044   Durbin-Watson:                     nan
Prob(Omnibus):                  0.593   Jarque-Bera (JB):               37.500
Skew:                           0.000   Prob(JB):                     7.19e-09
Kurtosis:                       0.000   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Summation x^2: 94.221754
Summation y^2: 94.221754

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                     y2   R-squared:                       0.005
Model:                            OLS   Adj. R-squared:                 -0.005
Method:                 Least Squares   F-statistic:                    0.5251
Date:                Fri, 04 Jan 2019   Prob (F-statistic):              0.470
Time:                        16:19:39   Log-Likelihood:                -138.65
No. Observations:                 100   AIC:                             279.3
Df Residuals:                      99   BIC:                             281.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x2             0.0726      0.100      0.725      0.470      -0.126       0.272
==============================================================================
Omnibus:                        0.504   Durbin-Watson:                   2.256
Prob(Omnibus):                  0.777   Jarque-Bera (JB):                0.303
Skew:                           0.133   Prob(JB):                        0.859
Kurtosis:                       3.043   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                     x2   R-squared:                       0.005
Model:                            OLS   Adj. R-squared:                 -0.005
Method:                 Least Squares   F-statistic:                    0.5251
Date:                Fri, 04 Jan 2019   Prob (F-statistic):              0.470
Time:                        16:19:39   Log-Likelihood:                -138.65
No. Observations:                 100   AIC:                             279.3
Df Residuals:                      99   BIC:                             281.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
y2             0.0726      0.100      0.725      0.470      -0.126       0.272
==============================================================================
Omnibus:                        0.480   Durbin-Watson:                   1.965
Prob(Omnibus):                  0.787   Jarque-Bera (JB):                0.220
Skew:                           0.104   Prob(JB):                        0.896
Kurtosis:                       3.096   Cond. No.                         1.00
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
<Figure size 792x360 with 0 Axes>

