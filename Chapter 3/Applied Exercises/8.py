#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:41:50 2018

@author: arpanganguli
"""

# import statistical tools
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# import and view first 10 items of file
url = "/Users/arpanganguli/Documents/Finance/ISLR/Datasets/Auto.csv"
Auto = pd.read_csv(url)
print(Auto.head(10))
print(list(Auto))

# run regression (I am not visualising the data for the sake of brevity. But it is highly recommended as first step afer importing data)
model = ols("Auto['mpg'] ~ Auto['horsepower']", data = Auto).fit()
print(model.summary())

"""
I was initially surprised to see the results given the number of parameters for horsepower in the solutions. But then I realised horsepower
is a categorical variable. This makes sense since automobiles come in various categories of horsepowers and is generally not continuous.This
simple overlook highlights the importance of viewing and understanding the nature of data before running any regression (although there
are a lot of other reasons why one should view and understand the data beforehand).
"""

"""
a. Given the F-Statistic > 1 and p-value of that F-Statistic is close to 0 (and << 0.005), there is a statistically significant
   relationship between mpg and horespower.
b. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
runfile('/Users/arpanganguli/Documents/Finance/ISLR/Codes - Python/Applied Exercises/8.py', wdir='/Users/arpanganguli/Documents/Finance/ISLR/Codes - Python/Applied Exercises')
    mpg  cylinders            ...              origin                       name
0  18.0          8            ...                   1  chevrolet chevelle malibu
1  15.0          8            ...                   1          buick skylark 320
2  18.0          8            ...                   1         plymouth satellite
3  16.0          8            ...                   1              amc rebel sst
4  17.0          8            ...                   1                ford torino
5  15.0          8            ...                   1           ford galaxie 500
6  14.0          8            ...                   1           chevrolet impala
7  14.0          8            ...                   1          plymouth fury iii
8  14.0          8            ...                   1           pontiac catalina
9  15.0          8            ...                   1         amc ambassador dpl

[10 rows x 9 columns]
['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            Auto['mpg']   R-squared:                       0.795
Model:                            OLS   Adj. R-squared:                  0.731
Method:                 Least Squares   F-statistic:                     12.60
Date:                Fri, 28 Dec 2018   Prob (F-statistic):           2.79e-64
Time:                        16:02:58   Log-Likelihood:                -1065.5
No. Observations:                 397   AIC:                             2319.
Df Residuals:                     303   BIC:                             2694.
Df Model:                          93                                         
Covariance Type:            nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                    19.5941      0.984     19.920      0.000      17.658      21.530
Auto['horsepower'][T.102]     0.4059      4.173      0.097      0.923      -7.806       8.618
Auto['horsepower'][T.103]     0.7059      4.173      0.169      0.866      -7.506       8.918
Auto['horsepower'][T.105]     0.9059      1.529      0.592      0.554      -2.103       3.915
Auto['horsepower'][T.107]     1.4059      4.173      0.337      0.736      -6.806       9.618
Auto['horsepower'][T.108]    -0.5941      4.173     -0.142      0.887      -8.806       7.618
Auto['horsepower'][T.110]     0.2392      1.372      0.174      0.862      -2.460       2.938
Auto['horsepower'][T.112]     0.0725      2.540      0.029      0.977      -4.925       5.070
Auto['horsepower'][T.113]     6.4059      4.173      1.535      0.126      -1.806      14.618
Auto['horsepower'][T.115]     5.1459      2.063      2.494      0.013       1.086       9.206
Auto['horsepower'][T.116]     5.8059      4.173      1.391      0.165      -2.406      14.018
Auto['horsepower'][T.120]    -1.0191      2.254     -0.452      0.651      -5.454       3.416
Auto['horsepower'][T.122]     0.4059      4.173      0.097      0.923      -7.806       8.618
Auto['horsepower'][T.125]     0.1392      2.540      0.055      0.956      -4.859       5.137
Auto['horsepower'][T.129]    -4.2941      3.032     -1.416      0.158     -10.260       1.672
Auto['horsepower'][T.130]    -4.3941      2.063     -2.130      0.034      -8.454      -0.334
Auto['horsepower'][T.132]    13.1059      4.173      3.140      0.002       4.894      21.318
Auto['horsepower'][T.133]    -3.3941      4.173     -0.813      0.417     -11.606       4.818
Auto['horsepower'][T.135]    -1.3941      4.173     -0.334      0.739      -9.606       6.818
Auto['horsepower'][T.137]    -5.5941      4.173     -1.340      0.181     -13.806       2.618
Auto['horsepower'][T.138]    -3.0941      4.173     -0.741      0.459     -11.306       5.118
Auto['horsepower'][T.139]    -0.4441      3.032     -0.146      0.884      -6.410       5.522
Auto['horsepower'][T.140]    -3.2513      1.821     -1.785      0.075      -6.835       0.333
Auto['horsepower'][T.142]    -4.0941      4.173     -0.981      0.327     -12.306       4.118
Auto['horsepower'][T.145]    -4.1370      1.821     -2.271      0.024      -7.721      -0.553
Auto['horsepower'][T.148]    -5.5941      4.173     -1.340      0.181     -13.806       2.618
Auto['horsepower'][T.149]    -3.5941      4.173     -0.861      0.390     -11.806       4.618
Auto['horsepower'][T.150]    -4.8896      1.310     -3.733      0.000      -7.467      -2.312
Auto['horsepower'][T.152]    -5.0941      4.173     -1.221      0.223     -13.306       3.118
Auto['horsepower'][T.153]    -5.5941      3.032     -1.845      0.066     -11.560       0.372
Auto['horsepower'][T.155]    -4.6441      3.032     -1.532      0.127     -10.610       1.322
Auto['horsepower'][T.158]    -6.5941      4.173     -1.580      0.115     -14.806       1.618
Auto['horsepower'][T.160]    -6.5941      3.032     -2.175      0.030     -12.560      -0.628
Auto['horsepower'][T.165]    -4.6691      2.254     -2.072      0.039      -9.104      -0.234
Auto['horsepower'][T.167]    -7.5941      4.173     -1.820      0.070     -15.806       0.618
Auto['horsepower'][T.170]    -5.0941      2.063     -2.469      0.014      -9.154      -1.034
Auto['horsepower'][T.175]    -6.1941      2.063     -3.002      0.003     -10.254      -2.134
Auto['horsepower'][T.180]    -6.0941      2.063     -2.954      0.003     -10.154      -2.034
Auto['horsepower'][T.190]    -5.0941      2.540     -2.006      0.046     -10.092      -0.096
Auto['horsepower'][T.193]   -10.5941      4.173     -2.539      0.012     -18.806      -2.382
Auto['horsepower'][T.198]    -6.0941      3.032     -2.010      0.045     -12.060      -0.128
Auto['horsepower'][T.200]    -9.5941      4.173     -2.299      0.022     -17.806      -1.382
Auto['horsepower'][T.208]    -8.5941      4.173     -2.059      0.040     -16.806      -0.382
Auto['horsepower'][T.210]    -8.5941      4.173     -2.059      0.040     -16.806      -0.382
Auto['horsepower'][T.215]    -7.2608      2.540     -2.859      0.005     -12.259      -2.263
Auto['horsepower'][T.220]    -5.5941      4.173     -1.340      0.181     -13.806       2.618
Auto['horsepower'][T.225]    -6.2608      2.540     -2.465      0.014     -11.259      -1.263
Auto['horsepower'][T.230]    -3.5941      4.173     -0.861      0.390     -11.806       4.618
Auto['horsepower'][T.46]      6.4059      3.032      2.113      0.035       0.440      12.372
Auto['horsepower'][T.48]     24.0059      2.540      9.452      0.000      19.008      29.004
Auto['horsepower'][T.49]      9.4059      4.173      2.254      0.025       1.194      17.618
Auto['horsepower'][T.52]     14.6059      2.254      6.481      0.000      10.171      19.041
Auto['horsepower'][T.53]     13.4059      3.032      4.422      0.000       7.440      19.372
Auto['horsepower'][T.54]      3.4059      4.173      0.816      0.415      -4.806      11.618
Auto['horsepower'][T.58]     17.9559      3.032      5.923      0.000      11.990      23.922
Auto['horsepower'][T.60]     12.5659      2.063      6.090      0.000       8.506      16.626
Auto['horsepower'][T.61]     12.4059      4.173      2.973      0.003       4.194      20.618
Auto['horsepower'][T.62]     14.1559      3.032      4.669      0.000       8.190      20.122
Auto['horsepower'][T.63]     14.8059      2.540      5.830      0.000       9.808      19.804
Auto['horsepower'][T.64]     19.4059      4.173      4.650      0.000      11.194      27.618
Auto['horsepower'][T.65]     15.8859      1.616      9.829      0.000      12.705      19.066
Auto['horsepower'][T.66]     16.5059      4.173      3.955      0.000       8.294      24.718
Auto['horsepower'][T.67]     13.9975      1.529      9.154      0.000      10.989      17.007
Auto['horsepower'][T.68]     12.5892      1.926      6.537      0.000       8.799      16.379
Auto['horsepower'][T.69]     13.1725      2.540      5.187      0.000       8.175      18.170
Auto['horsepower'][T.70]     12.8809      1.529      8.424      0.000       9.872      15.890
Auto['horsepower'][T.71]      9.4259      2.063      4.568      0.000       5.366      13.486
Auto['horsepower'][T.72]      2.3892      1.926      1.241      0.216      -1.401       6.179
Auto['horsepower'][T.74]     13.9392      2.540      5.488      0.000       8.941      18.937
Auto['horsepower'][T.75]      9.4416      1.464      6.451      0.000       6.561      12.322
Auto['horsepower'][T.76]     11.4559      2.254      5.083      0.000       7.021      15.891
Auto['horsepower'][T.77]      5.8059      4.173      1.391      0.165      -2.406      14.018
Auto['horsepower'][T.78]      7.2059      1.926      3.742      0.000       3.416      10.996
Auto['horsepower'][T.79]      7.4059      3.032      2.443      0.015       1.440      13.372
Auto['horsepower'][T.80]      9.0059      1.821      4.945      0.000       5.422      12.590
Auto['horsepower'][T.81]      4.9059      3.032      1.618      0.107      -1.060      10.872
Auto['horsepower'][T.82]     11.4059      4.173      2.733      0.007       3.194      19.618
Auto['horsepower'][T.83]      8.5309      2.254      3.785      0.000       4.096      12.966
Auto['horsepower'][T.84]     10.5392      1.926      5.473      0.000       6.749      14.329
Auto['horsepower'][T.85]      3.8725      1.672      2.316      0.021       0.583       7.162
Auto['horsepower'][T.86]      4.6059      2.063      2.232      0.026       0.546       8.666
Auto['horsepower'][T.87]      3.4059      3.032      1.123      0.262      -2.560       9.372
Auto['horsepower'][T.88]      5.4848      1.354      4.051      0.000       2.820       8.149
Auto['horsepower'][T.89]      5.9059      4.173      1.415      0.158      -2.306      14.118
Auto['horsepower'][T.90]      4.7609      1.338      3.559      0.000       2.128       7.394
Auto['horsepower'][T.91]      0.4059      4.173      0.097      0.923      -7.806       8.618
Auto['horsepower'][T.92]      8.0392      1.926      4.174      0.000       4.249      11.829
Auto['horsepower'][T.93]      6.4059      4.173      1.535      0.126      -1.806      14.618
Auto['horsepower'][T.94]      2.4059      4.173      0.577      0.565      -5.806      10.618
Auto['horsepower'][T.95]      2.5202      1.464      1.722      0.086      -0.360       5.400
Auto['horsepower'][T.96]      7.5725      2.540      2.982      0.003       2.575      12.570
Auto['horsepower'][T.97]      2.5281      1.672      1.512      0.132      -0.762       5.818
Auto['horsepower'][T.98]      0.6559      3.032      0.216      0.829      -5.310       6.622
Auto['horsepower'][T.?]       9.4059      2.063      4.559      0.000       5.346      13.466
==============================================================================
Omnibus:                       52.669   Durbin-Watson:                   1.388
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               95.031
Skew:                           0.773   Prob(JB):                     2.31e-21
Kurtosis:                       4.831   Cond. No.                         49.7
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
