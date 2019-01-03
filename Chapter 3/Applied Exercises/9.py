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

# 9.c. Run multivariate regression

Auto['hp'] = Auto['horsepower'].astype(float)
"""
For some annoying reason, Python is importing the horsepower
column as string and not float. This will impact the regression
results since we cannot regress string values. So, I am converting
this column into float and storing the values in to a new column
called "hp". I will use the values in "hp" to regress "mpg".
"""
X = Auto[['cylinders', 'displacement', 'hp', 'weight',
       'acceleration', 'year', 'origin']]
Y = Auto['mpg']
X1 = sm.add_constant(X)
reg = sm.OLS(Y, X1).fit()
print(reg.summary())

# 9.e. Run multivariate regression with interaction terms

Auto['hp'] = Auto['horsepower'].astype(float)
"""
For some annoying reason, Python is importing the horsepower
column as string and not float. This will impact the regression
results since we cannot regress string values. So, I am converting
this column into float and storing the values in to a new column
called "hp". I will use the values in "hp" to regress "mpg".
"""

X1 = Auto['hp']
X2 = Auto['weight']
X3 = Auto['acceleration']
X4 = Auto['year']
X5 = Auto['origin']
X6 = Auto['displacement']
X7 = Auto['cylinders']
Y = Auto['mpg']

reg = ols("Y~X1+X2+X3+X4+X5+X6+X7+I(np.log(X1))+I(X4^2)", data = Auto).fit()
"""
I randomly chose two transformations for two variables:
    1.  Log-transformation for X1: OLS result suggests that for a unit
        change in log(X1), the miles per gallon reduces by ~27.2 units
    2. Square of X4: OLS result suggests that for a unit increase in 
        X4^2, the miles per gallon reduces by 0.12 units. However, the
        high p-value of this statistic suggests that the null hypothesis
        cannot be rejected. Therefore, essentially there is no difference
        between this particular value and 0, and therefore this statistic
        can be discarded.
        
"""
print(reg.summary())
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
runfile('/Users/arpanganguli/.spyder-py3/temp.py', wdir='/Users/arpanganguli/.spyder-py3')
    mpg  cylinders            ...              origin                       name
0  18.0          8            ...                   1  chevrolet chevelle malibu
1  15.0          8            ...                   1          buick skylark 320
2  18.0          8            ...                   1         plymouth satellite
3  16.0          8            ...                   1              amc rebel sst
4  17.0          8            ...                   1                ford torino

[5 rows x 9 columns]
['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
<class 'pandas.core.frame.DataFrame'>
Int64Index: 392 entries, 0 to 396
Data columns (total 9 columns):
mpg             392 non-null float64
cylinders       392 non-null int64
displacement    392 non-null float64
horsepower      392 non-null object
weight          392 non-null int64
acceleration    392 non-null float64
year            392 non-null int64
origin          392 non-null int64
name            392 non-null object
dtypes: float64(3), int64(4), object(2)
memory usage: 30.6+ KB
None
                   mpg  cylinders    ...         year    origin
mpg           1.000000  -0.777618    ...     0.580541  0.565209
cylinders    -0.777618   1.000000    ...    -0.345647 -0.568932
displacement -0.805127   0.950823    ...    -0.369855 -0.614535
weight       -0.832244   0.897527    ...    -0.309120 -0.585005
acceleration  0.423329  -0.504683    ...     0.290316  0.212746
year          0.580541  -0.345647    ...     1.000000  0.181528
origin        0.565209  -0.568932    ...     0.181528  1.000000

[7 rows x 7 columns]
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.821
Model:                            OLS   Adj. R-squared:                  0.818
Method:                 Least Squares   F-statistic:                     252.4
Date:                Wed, 02 Jan 2019   Prob (F-statistic):          2.04e-139
Time:                        21:50:38   Log-Likelihood:                -1023.5
No. Observations:                 392   AIC:                             2063.
Df Residuals:                     384   BIC:                             2095.
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const          -17.2184      4.644     -3.707      0.000     -26.350      -8.087
cylinders       -0.4934      0.323     -1.526      0.128      -1.129       0.142
displacement     0.0199      0.008      2.647      0.008       0.005       0.035
hp              -0.0170      0.014     -1.230      0.220      -0.044       0.010
weight          -0.0065      0.001     -9.929      0.000      -0.008      -0.005
acceleration     0.0806      0.099      0.815      0.415      -0.114       0.275
year             0.7508      0.051     14.729      0.000       0.651       0.851
origin           1.4261      0.278      5.127      0.000       0.879       1.973
==============================================================================
Omnibus:                       31.906   Durbin-Watson:                   1.309
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               53.100
Skew:                           0.529   Prob(JB):                     2.95e-12
Kurtosis:                       4.460   Cond. No.                     8.59e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 8.59e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

 OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.859
Model:                            OLS   Adj. R-squared:                  0.855
Method:                 Least Squares   F-statistic:                     232.0
Date:                Thu, 03 Jan 2019   Prob (F-statistic):          2.98e-155
Time:                        16:36:22   Log-Likelihood:                -977.34
No. Observations:                 392   AIC:                             1977.
Df Residuals:                     381   BIC:                             2020.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -6.8302      6.090     -1.122      0.263     -18.805       5.144
X1            -0.0355      0.013     -2.693      0.007      -0.061      -0.010
X2            -0.0093      0.002     -3.929      0.000      -0.014      -0.005
X3             0.0603      0.089      0.677      0.499      -0.115       0.235
X4             0.7831      0.046     17.120      0.000       0.693       0.873
X5             0.5193      0.271      1.919      0.056      -0.013       1.051
X6            -0.0859      0.031     -2.738      0.006      -0.148      -0.024
X7             0.5961      1.536      0.388      0.698      -2.423       3.616
X7:X6          0.0019      0.003      0.622      0.535      -0.004       0.008
X7:X2         -0.0003      0.001     -0.498      0.618      -0.001       0.001
X6:X2       2.386e-05   6.16e-06      3.872      0.000    1.17e-05     3.6e-05
==============================================================================
Omnibus:                       45.965   Durbin-Watson:                   1.409
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               99.574
Skew:                           0.628   Prob(JB):                     2.39e-22
Kurtosis:                       5.125   Cond. No.                     3.48e+07
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.48e+07. This might indicate that there are
strong multicollinearity or other numerical problems.

<Auto - Scatterplot Matrix.png>
