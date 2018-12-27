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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RESULTS (NOT PART OF CODE)

runfile('/Users/arpanganguli/Documents/Finance : ML/ISLR/Codes - Python/Section3.6.2.py', wdir='/Users/arpanganguli/Documents/Finance : ML/ISLR/Codes - Python')
      crim    zn  indus  chas    nox  ...   tax  ptratio   black  lstat  medv
1  0.00632  18.0   2.31     0  0.538  ...   296     15.3  396.90   4.98  24.0
2  0.02731   0.0   7.07     0  0.469  ...   242     17.8  396.90   9.14  21.6
3  0.02729   0.0   7.07     0  0.469  ...   242     17.8  392.83   4.03  34.7
4  0.03237   0.0   2.18     0  0.458  ...   222     18.7  394.63   2.94  33.4
5  0.06905   0.0   2.18     0  0.458  ...   222     18.7  396.90   5.33  36.2

[5 rows x 14 columns]
             crim        zn     indus    ...        black     lstat      medv
crim     1.000000 -0.200469  0.406583    ...    -0.385064  0.455621 -0.388305
zn      -0.200469  1.000000 -0.533828    ...     0.175520 -0.412995  0.360445
indus    0.406583 -0.533828  1.000000    ...    -0.356977  0.603800 -0.483725
chas    -0.055892 -0.042697  0.062938    ...     0.048788 -0.053929  0.175260
nox      0.420972 -0.516604  0.763651    ...    -0.380051  0.590879 -0.427321
rm      -0.219247  0.311991 -0.391676    ...     0.128069 -0.613808  0.695360
age      0.352734 -0.569537  0.644779    ...    -0.273534  0.602339 -0.376955
dis     -0.379670  0.664408 -0.708027    ...     0.291512 -0.496996  0.249929
rad      0.625505 -0.311948  0.595129    ...    -0.444413  0.488676 -0.381626
tax      0.582764 -0.314563  0.720760    ...    -0.441808  0.543993 -0.468536
ptratio  0.289946 -0.391679  0.383248    ...    -0.177383  0.374044 -0.507787
black   -0.385064  0.175520 -0.356977    ...     1.000000 -0.366087  0.333461
lstat    0.455621 -0.412995  0.603800    ...    -0.366087  1.000000 -0.737663
medv    -0.388305  0.360445 -0.483725    ...     0.333461 -0.737663  1.000000

[14 rows x 14 columns]
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.544
Model:                            OLS   Adj. R-squared:                  0.543
Method:                 Least Squares   F-statistic:                     601.6
Date:                Thu, 27 Dec 2018   Prob (F-statistic):           5.08e-88
Time:                        18:04:49   Log-Likelihood:                -1641.5
No. Observations:                 506   AIC:                             3287.
Df Residuals:                     504   BIC:                             3295.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     34.5538      0.563     61.415      0.000      33.448      35.659
X             -0.9500      0.039    -24.528      0.000      -1.026      -0.874
==============================================================================
Omnibus:                      137.043   Durbin-Watson:                   0.892
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              291.373
Skew:                           1.453   Prob(JB):                     5.36e-64
Kurtosis:                       5.319   Cond. No.                         29.7
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
           0          1          2
0  29.822595  25.618169  34.027022
1  25.870390  21.669748  30.071032
2  30.725142  26.519457  34.930827
3  31.760696  27.553387  35.968004
4  29.490078  25.286078  33.694078
AxesSubplot(0.125,0.125;0.352273x0.755)
/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
AxesSubplot(0.547727,0.125;0.352273x0.755)

<Boston - Regression and Residuals.png>
