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
from scipy import stats

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

# import data
url = "/Users/arpanganguli/Documents/Finance/ISLR/Datasets/Boston.csv"
Boston = pd.read_csv(url, index_col = 'SlNo')

# run simple linear regressions for each independent variable
Boston_columns = list(Boston)
for t in Boston_columns:
    reg = ols("crim~Boston[t]", data = Boston).fit()
    print(reg.summary())
    print()
    plt.xkcd()
    plt.figure(figsize = (11, 5))
    sns.regplot(reg.predict(), reg.resid, data = Boston)
    plt.title(t)
    plt.xlabel(t)
    plt.ylabel('crim')
    print()
    print(colored("="*78, 'green'))
    print()

# run multivariate linear regression for 'crim'
print(list(Boston))
X1 = Boston.iloc[:,[1,2,4,5,6,7,8, 9,10,11,12,13]]
X2 = Boston['chas']
reg = ols("crim~zn+indus+C(chas)+nox+rm+age+dis+rad\
          +tax+ptratio+black+lstat+medv", data = Boston).fit()
print(reg.summary())
print()
print(colored("="*78, 'green'))
print()
plt.xkcd()
sns.pairplot(Boston)
plt.title("Boston Pairplot")

# run polynomial regressions for each independent variable
Boston_columns = list(Boston)
for t in Boston_columns:
    reg = ols("crim~Boston[t]+I(pow(Boston[t],2)) +\
              I(pow(Boston[t],3))", data = Boston).fit()
    print(reg.summary())
    print()
    print(colored("="*78, 'green'))
    print()


sns.dogplot() # displays a dog's photo. What's life without some fun?!

"""
15.a., b. & c. There are statistically significant association between the predictor
and response for 'dis', 'rad', 'black', 'medv' in the multivariate linear
regression model.

15.d. The answer is 'yes' for all but 'black' and 'chas'.
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESULTS (NOT PART OF CODE)

runfile('/Users/arpanganguli/Documents/Finance/ISLR/General_Code.py', wdir='/Users/arpanganguli/Documents/Finance/ISLR')
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 2.207e+33
Date:                Sat, 05 Jan 2019   Prob (F-statistic):               0.00
Time:                        01:00:27   Log-Likelihood:                 16044.
No. Observations:                 506   AIC:                        -3.208e+04
Df Residuals:                     504   BIC:                        -3.208e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept  -2.082e-16   1.98e-16     -1.049      0.295   -5.98e-16    1.82e-16
Boston[t]      1.0000   2.13e-17    4.7e+16      0.000       1.000       1.000
==============================================================================
Omnibus:                      587.527   Durbin-Watson:                   0.672
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            40133.940
Skew:                           5.509   Prob(JB):                         0.00
Kurtosis:                      45.216   Cond. No.                         10.1
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.040
Model:                            OLS   Adj. R-squared:                  0.038
Method:                 Least Squares   F-statistic:                     21.10
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           5.51e-06
Time:                        01:00:27   Log-Likelihood:                -1796.0
No. Observations:                 506   AIC:                             3596.
Df Residuals:                     504   BIC:                             3604.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      4.4537      0.417     10.675      0.000       3.634       5.273
Boston[t]     -0.0739      0.016     -4.594      0.000      -0.106      -0.042
==============================================================================
Omnibus:                      567.443   Durbin-Watson:                   0.857
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            32753.004
Skew:                           5.257   Prob(JB):                         0.00
Kurtosis:                      40.986   Cond. No.                         28.8
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.165
Model:                            OLS   Adj. R-squared:                  0.164
Method:                 Least Squares   F-statistic:                     99.82
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.45e-21
Time:                        01:00:27   Log-Likelihood:                -1760.6
No. Observations:                 506   AIC:                             3525.
Df Residuals:                     504   BIC:                             3534.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -2.0637      0.667     -3.093      0.002      -3.375      -0.753
Boston[t]      0.5098      0.051      9.991      0.000       0.410       0.610
==============================================================================
Omnibus:                      585.118   Durbin-Watson:                   0.986
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            41418.938
Skew:                           5.449   Prob(JB):                         0.00
Kurtosis:                      45.962   Cond. No.                         25.1
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.003
Model:                            OLS   Adj. R-squared:                  0.001
Method:                 Least Squares   F-statistic:                     1.579
Date:                Sat, 05 Jan 2019   Prob (F-statistic):              0.209
Time:                        01:00:27   Log-Likelihood:                -1805.6
No. Observations:                 506   AIC:                             3615.
Df Residuals:                     504   BIC:                             3624.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      3.7444      0.396      9.453      0.000       2.966       4.523
Boston[t]     -1.8928      1.506     -1.257      0.209      -4.852       1.066
==============================================================================
Omnibus:                      561.663   Durbin-Watson:                   0.817
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            30645.429
Skew:                           5.191   Prob(JB):                         0.00
Kurtosis:                      39.685   Cond. No.                         3.96
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.177
Model:                            OLS   Adj. R-squared:                  0.176
Method:                 Least Squares   F-statistic:                     108.6
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           3.75e-23
Time:                        01:00:28   Log-Likelihood:                -1757.0
No. Observations:                 506   AIC:                             3518.
Df Residuals:                     504   BIC:                             3526.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -13.7199      1.699     -8.073      0.000     -17.059     -10.381
Boston[t]     31.2485      2.999     10.419      0.000      25.356      37.141
==============================================================================
Omnibus:                      591.712   Durbin-Watson:                   0.992
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            43138.106
Skew:                           5.546   Prob(JB):                         0.00
Kurtosis:                      46.852   Cond. No.                         11.3
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.048
Model:                            OLS   Adj. R-squared:                  0.046
Method:                 Least Squares   F-statistic:                     25.45
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           6.35e-07
Time:                        01:00:28   Log-Likelihood:                -1793.9
No. Observations:                 506   AIC:                             3592.
Df Residuals:                     504   BIC:                             3600.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     20.4818      3.364      6.088      0.000      13.872      27.092
Boston[t]     -2.6841      0.532     -5.045      0.000      -3.729      -1.639
==============================================================================
Omnibus:                      575.717   Durbin-Watson:                   0.879
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            36658.093
Skew:                           5.345   Prob(JB):                         0.00
Kurtosis:                      43.305   Cond. No.                         58.4
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.124
Model:                            OLS   Adj. R-squared:                  0.123
Method:                 Least Squares   F-statistic:                     71.62
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           2.85e-16
Time:                        01:00:28   Log-Likelihood:                -1772.7
No. Observations:                 506   AIC:                             3549.
Df Residuals:                     504   BIC:                             3558.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.7779      0.944     -4.002      0.000      -5.633      -1.923
Boston[t]      0.1078      0.013      8.463      0.000       0.083       0.133
==============================================================================
Omnibus:                      574.509   Durbin-Watson:                   0.956
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            36741.903
Skew:                           5.322   Prob(JB):                         0.00
Kurtosis:                      43.366   Cond. No.                         195.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.144
Model:                            OLS   Adj. R-squared:                  0.142
Method:                 Least Squares   F-statistic:                     84.89
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           8.52e-19
Time:                        01:00:28   Log-Likelihood:                -1767.0
No. Observations:                 506   AIC:                             3538.
Df Residuals:                     504   BIC:                             3546.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      9.4993      0.730     13.006      0.000       8.064      10.934
Boston[t]     -1.5509      0.168     -9.213      0.000      -1.882      -1.220
==============================================================================
Omnibus:                      576.519   Durbin-Watson:                   0.952
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            37426.729
Skew:                           5.348   Prob(JB):                         0.00
Kurtosis:                      43.753   Cond. No.                         9.32
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.391
Model:                            OLS   Adj. R-squared:                  0.390
Method:                 Least Squares   F-statistic:                     323.9
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           2.69e-56
Time:                        01:00:29   Log-Likelihood:                -1680.8
No. Observations:                 506   AIC:                             3366.
Df Residuals:                     504   BIC:                             3374.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -2.2872      0.443     -5.157      0.000      -3.158      -1.416
Boston[t]      0.6179      0.034     17.998      0.000       0.550       0.685
==============================================================================
Omnibus:                      656.459   Durbin-Watson:                   1.337
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            75417.007
Skew:                           6.478   Prob(JB):                         0.00
Kurtosis:                      61.389   Cond. No.                         19.2
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.340
Model:                            OLS   Adj. R-squared:                  0.338
Method:                 Least Squares   F-statistic:                     259.2
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           2.36e-47
Time:                        01:00:29   Log-Likelihood:                -1701.4
No. Observations:                 506   AIC:                             3407.
Df Residuals:                     504   BIC:                             3415.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -8.5284      0.816    -10.454      0.000     -10.131      -6.926
Boston[t]      0.0297      0.002     16.099      0.000       0.026       0.033
==============================================================================
Omnibus:                      635.377   Durbin-Watson:                   1.252
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            63763.835
Skew:                           6.156   Prob(JB):                         0.00
Kurtosis:                      56.599   Cond. No.                     1.16e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.16e+03. This might indicate that there are
strong multicollinearity or other numerical problems.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.084
Model:                            OLS   Adj. R-squared:                  0.082
Method:                 Least Squares   F-statistic:                     46.26
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           2.94e-11
Time:                        01:00:29   Log-Likelihood:                -1784.1
No. Observations:                 506   AIC:                             3572.
Df Residuals:                     504   BIC:                             3581.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -17.6469      3.147     -5.607      0.000     -23.830     -11.464
Boston[t]      1.1520      0.169      6.801      0.000       0.819       1.485
==============================================================================
Omnibus:                      568.053   Durbin-Watson:                   0.905
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            34221.853
Skew:                           5.245   Prob(JB):                         0.00
Kurtosis:                      41.899   Cond. No.                         160.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.148
Model:                            OLS   Adj. R-squared:                  0.147
Method:                 Least Squares   F-statistic:                     87.74
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           2.49e-19
Time:                        01:00:29   Log-Likelihood:                -1765.8
No. Observations:                 506   AIC:                             3536.
Df Residuals:                     504   BIC:                             3544.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     16.5535      1.426     11.609      0.000      13.752      19.355
Boston[t]     -0.0363      0.004     -9.367      0.000      -0.044      -0.029
==============================================================================
Omnibus:                      594.029   Durbin-Watson:                   0.994
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            44041.935
Skew:                           5.578   Prob(JB):                         0.00
Kurtosis:                      47.323   Cond. No.                     1.49e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.49e+03. This might indicate that there are
strong multicollinearity or other numerical problems.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.208
Model:                            OLS   Adj. R-squared:                  0.206
Method:                 Least Squares   F-statistic:                     132.0
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           2.65e-27
Time:                        01:00:30   Log-Likelihood:                -1747.5
No. Observations:                 506   AIC:                             3499.
Df Residuals:                     504   BIC:                             3507.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.3305      0.694     -4.801      0.000      -4.694      -1.968
Boston[t]      0.5488      0.048     11.491      0.000       0.455       0.643
==============================================================================
Omnibus:                      601.306   Durbin-Watson:                   1.182
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            49918.826
Skew:                           5.645   Prob(JB):                         0.00
Kurtosis:                      50.331   Cond. No.                         29.7
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.151
Model:                            OLS   Adj. R-squared:                  0.149
Method:                 Least Squares   F-statistic:                     89.49
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.17e-19
Time:                        01:00:30   Log-Likelihood:                -1765.0
No. Observations:                 506   AIC:                             3534.
Df Residuals:                     504   BIC:                             3542.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     11.7965      0.934     12.628      0.000       9.961      13.632
Boston[t]     -0.3632      0.038     -9.460      0.000      -0.439      -0.288
==============================================================================
Omnibus:                      558.880   Durbin-Watson:                   0.996
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            32740.044
Skew:                           5.108   Prob(JB):                         0.00
Kurtosis:                      41.059   Cond. No.                         64.5
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


==============================================================================

['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.454
Model:                            OLS   Adj. R-squared:                  0.440
Method:                 Least Squares   F-statistic:                     31.47
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.57e-56
Time:                        01:00:30   Log-Likelihood:                -1653.3
No. Observations:                 506   AIC:                             3335.
Df Residuals:                     492   BIC:                             3394.
Df Model:                          13                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       17.0332      7.235      2.354      0.019       2.818      31.248
C(chas)[T.1]    -0.7491      1.180     -0.635      0.526      -3.068       1.570
zn               0.0449      0.019      2.394      0.017       0.008       0.082
indus           -0.0639      0.083     -0.766      0.444      -0.228       0.100
nox            -10.3135      5.276     -1.955      0.051     -20.679       0.052
rm               0.4301      0.613      0.702      0.483      -0.774       1.634
age              0.0015      0.018      0.081      0.935      -0.034       0.037
dis             -0.9872      0.282     -3.503      0.001      -1.541      -0.433
rad              0.5882      0.088      6.680      0.000       0.415       0.761
tax             -0.0038      0.005     -0.733      0.464      -0.014       0.006
ptratio         -0.2711      0.186     -1.454      0.147      -0.637       0.095
black           -0.0075      0.004     -2.052      0.041      -0.015      -0.000
lstat            0.1262      0.076      1.667      0.096      -0.023       0.275
medv            -0.1989      0.061     -3.287      0.001      -0.318      -0.080
==============================================================================
Omnibus:                      666.613   Durbin-Watson:                   1.519
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            84887.625
Skew:                           6.617   Prob(JB):                         0.00
Kurtosis:                      65.058   Cond. No.                     1.58e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.58e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 4.285e+27
Date:                Sat, 05 Jan 2019   Prob (F-statistic):               0.00
Time:                        01:12:00   Log-Likelihood:                 12995.
No. Observations:                 506   AIC:                        -2.598e+04
Df Residuals:                     502   BIC:                        -2.597e+04
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept             1.636e-12    9.1e-14     17.978      0.000    1.46e-12    1.81e-12
Boston[t]                1.0000   3.02e-14   3.31e+13      0.000       1.000       1.000
I(pow(Boston[t], 2))  7.321e-15   1.42e-15      5.146      0.000    4.53e-15    1.01e-14
I(pow(Boston[t], 3)) -6.825e-17   1.39e-17     -4.919      0.000   -9.55e-17    -4.1e-17
==============================================================================
Omnibus:                      529.776   Durbin-Watson:                   0.446
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            32923.102
Skew:                           4.617   Prob(JB):                         0.00
Kurtosis:                      41.423   Cond. No.                     4.82e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.82e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.058
Model:                            OLS   Adj. R-squared:                  0.053
Method:                 Least Squares   F-statistic:                     10.35
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.28e-06
Time:                        01:12:00   Log-Likelihood:                -1791.2
No. Observations:                 506   AIC:                             3590.
Df Residuals:                     502   BIC:                             3607.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept                4.8461      0.433     11.192      0.000       3.995       5.697
Boston[t]               -0.3322      0.110     -3.025      0.003      -0.548      -0.116
I(pow(Boston[t], 2))     0.0065      0.004      1.679      0.094      -0.001       0.014
I(pow(Boston[t], 3)) -3.776e-05   3.14e-05     -1.203      0.230   -9.94e-05    2.39e-05
==============================================================================
Omnibus:                      569.133   Durbin-Watson:                   0.875
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            33700.991
Skew:                           5.272   Prob(JB):                         0.00
Kurtosis:                      41.565   Cond. No.                     1.89e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.89e+05. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.260
Model:                            OLS   Adj. R-squared:                  0.255
Method:                 Least Squares   F-statistic:                     58.69
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.55e-32
Time:                        01:12:00   Log-Likelihood:                -1730.3
No. Observations:                 506   AIC:                             3469.
Df Residuals:                     502   BIC:                             3486.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept                3.6626      1.574      2.327      0.020       0.570       6.755
Boston[t]               -1.9652      0.482     -4.077      0.000      -2.912      -1.018
I(pow(Boston[t], 2))     0.2519      0.039      6.407      0.000       0.175       0.329
I(pow(Boston[t], 3))    -0.0070      0.001     -7.292      0.000      -0.009      -0.005
==============================================================================
Omnibus:                      611.788   Durbin-Watson:                   1.116
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            51742.286
Skew:                           5.820   Prob(JB):                         0.00
Kurtosis:                      51.153   Cond. No.                     2.47e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.47e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.003
Model:                            OLS   Adj. R-squared:                  0.001
Method:                 Least Squares   F-statistic:                     1.579
Date:                Sat, 05 Jan 2019   Prob (F-statistic):              0.209
Time:                        01:12:00   Log-Likelihood:                -1805.6
No. Observations:                 506   AIC:                             3615.
Df Residuals:                     504   BIC:                             3624.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept                3.7444      0.396      9.453      0.000       2.966       4.523
Boston[t]               -0.6309      0.502     -1.257      0.209      -1.617       0.355
I(pow(Boston[t], 2))    -0.6309      0.502     -1.257      0.209      -1.617       0.355
I(pow(Boston[t], 3))    -0.6309      0.502     -1.257      0.209      -1.617       0.355
==============================================================================
Omnibus:                      561.663   Durbin-Watson:                   0.817
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            30645.429
Skew:                           5.191   Prob(JB):                         0.00
Kurtosis:                      39.685   Cond. No.                     3.42e+32
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 4.39e-63. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.297
Model:                            OLS   Adj. R-squared:                  0.293
Method:                 Least Squares   F-statistic:                     70.69
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           3.81e-38
Time:                        01:12:00   Log-Likelihood:                -1717.2
No. Observations:                 506   AIC:                             3442.
Df Residuals:                     502   BIC:                             3459.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept              233.0866     33.643      6.928      0.000     166.988     299.185
Boston[t]            -1279.3713    170.397     -7.508      0.000   -1614.151    -944.591
I(pow(Boston[t], 2))  2248.5441    279.899      8.033      0.000    1698.626    2798.462
I(pow(Boston[t], 3)) -1245.7029    149.282     -8.345      0.000   -1538.997    -952.409
==============================================================================
Omnibus:                      614.412   Durbin-Watson:                   1.159
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            53523.997
Skew:                           5.851   Prob(JB):                         0.00
Kurtosis:                      52.008   Cond. No.                     1.36e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.36e+03. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.068
Model:                            OLS   Adj. R-squared:                  0.062
Method:                 Least Squares   F-statistic:                     12.17
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.07e-07
Time:                        01:12:00   Log-Likelihood:                -1788.6
No. Observations:                 506   AIC:                             3585.
Df Residuals:                     502   BIC:                             3602.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept              112.6246     64.517      1.746      0.081     -14.132     239.382
Boston[t]              -39.1501     31.311     -1.250      0.212    -100.668      22.368
I(pow(Boston[t], 2))     4.5509      5.010      0.908      0.364      -5.292      14.394
I(pow(Boston[t], 3))    -0.1745      0.264     -0.662      0.509      -0.693       0.344
==============================================================================
Omnibus:                      585.097   Durbin-Watson:                   0.913
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            40144.207
Skew:                           5.465   Prob(JB):                         0.00
Kurtosis:                      45.245   Cond. No.                     5.36e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.36e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.174
Model:                            OLS   Adj. R-squared:                  0.169
Method:                 Least Squares   F-statistic:                     35.31
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.02e-20
Time:                        01:12:00   Log-Likelihood:                -1757.9
No. Observations:                 506   AIC:                             3524.
Df Residuals:                     502   BIC:                             3541.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -2.5488      2.769     -0.920      0.358      -7.989       2.892
Boston[t]                0.2737      0.186      1.468      0.143      -0.093       0.640
I(pow(Boston[t], 2))    -0.0072      0.004     -1.988      0.047      -0.014    -8.4e-05
I(pow(Boston[t], 3))  5.745e-05   2.11e-05      2.724      0.007     1.6e-05    9.89e-05
==============================================================================
Omnibus:                      577.477   Durbin-Watson:                   1.025
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            39586.670
Skew:                           5.336   Prob(JB):                         0.00
Kurtosis:                      44.997   Cond. No.                     4.74e+06
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.74e+06. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.278
Model:                            OLS   Adj. R-squared:                  0.274
Method:                 Least Squares   F-statistic:                     64.37
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           3.14e-35
Time:                        01:12:01   Log-Likelihood:                -1724.0
No. Observations:                 506   AIC:                             3456.
Df Residuals:                     502   BIC:                             3473.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               30.0476      2.446     12.285      0.000      25.242      34.853
Boston[t]              -15.5544      1.736     -8.960      0.000     -18.965     -12.144
I(pow(Boston[t], 2))     2.4521      0.346      7.078      0.000       1.771       3.133
I(pow(Boston[t], 3))    -0.1186      0.020     -5.814      0.000      -0.159      -0.079
==============================================================================
Omnibus:                      577.742   Durbin-Watson:                   1.129
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            42444.706
Skew:                           5.305   Prob(JB):                         0.00
Kurtosis:                      46.596   Cond. No.                     2.10e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.1e+03. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.400
Model:                            OLS   Adj. R-squared:                  0.396
Method:                 Least Squares   F-statistic:                     111.6
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           2.31e-55
Time:                        01:12:01   Log-Likelihood:                -1677.1
No. Observations:                 506   AIC:                             3362.
Df Residuals:                     502   BIC:                             3379.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -0.6055      2.050     -0.295      0.768      -4.633       3.422
Boston[t]                0.5127      1.044      0.491      0.623      -1.538       2.563
I(pow(Boston[t], 2))    -0.0752      0.149     -0.506      0.613      -0.367       0.217
I(pow(Boston[t], 3))     0.0032      0.005      0.703      0.482      -0.006       0.012
==============================================================================
Omnibus:                      659.751   Durbin-Watson:                   1.351
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            77838.247
Skew:                           6.526   Prob(JB):                         0.00
Kurtosis:                      62.343   Cond. No.                     5.43e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.43e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.369
Model:                            OLS   Adj. R-squared:                  0.365
Method:                 Least Squares   F-statistic:                     97.80
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           7.34e-50
Time:                        01:12:01   Log-Likelihood:                -1689.9
No. Observations:                 506   AIC:                             3388.
Df Residuals:                     502   BIC:                             3405.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               19.1836     11.796      1.626      0.105      -3.991      42.358
Boston[t]               -0.1533      0.096     -1.602      0.110      -0.341       0.035
I(pow(Boston[t], 2))     0.0004      0.000      1.488      0.137      -0.000       0.001
I(pow(Boston[t], 3)) -2.204e-07   1.89e-07     -1.167      0.244   -5.91e-07    1.51e-07
==============================================================================
Omnibus:                      644.161   Durbin-Watson:                   1.293
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            69773.212
Skew:                           6.278   Prob(JB):                         0.00
Kurtosis:                      59.141   Cond. No.                     6.16e+09
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 6.16e+09. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.114
Model:                            OLS   Adj. R-squared:                  0.108
Method:                 Least Squares   F-statistic:                     21.48
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           4.17e-13
Time:                        01:12:01   Log-Likelihood:                -1775.8
No. Observations:                 506   AIC:                             3560.
Df Residuals:                     502   BIC:                             3577.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept              477.1840    156.795      3.043      0.002     169.129     785.239
Boston[t]              -82.3605     27.644     -2.979      0.003    -136.673     -28.048
I(pow(Boston[t], 2))     4.6353      1.608      2.882      0.004       1.475       7.795
I(pow(Boston[t], 3))    -0.0848      0.031     -2.743      0.006      -0.145      -0.024
==============================================================================
Omnibus:                      572.356   Durbin-Watson:                   0.945
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            36070.763
Skew:                           5.294   Prob(JB):                         0.00
Kurtosis:                      42.985   Cond. No.                     3.02e+06
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.02e+06. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.150
Model:                            OLS   Adj. R-squared:                  0.145
Method:                 Least Squares   F-statistic:                     29.49
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.41e-17
Time:                        01:12:01   Log-Likelihood:                -1765.3
No. Observations:                 506   AIC:                             3539.
Df Residuals:                     502   BIC:                             3555.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               18.2637      2.305      7.924      0.000      13.735      22.792
Boston[t]               -0.0836      0.056     -1.483      0.139      -0.194       0.027
I(pow(Boston[t], 2))     0.0002      0.000      0.716      0.474      -0.000       0.001
I(pow(Boston[t], 3)) -2.652e-07   4.36e-07     -0.608      0.544   -1.12e-06    5.92e-07
==============================================================================
Omnibus:                      591.816   Durbin-Watson:                   0.983
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            43468.746
Skew:                           5.544   Prob(JB):                         0.00
Kurtosis:                      47.032   Cond. No.                     3.59e+08
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.59e+08. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.218
Model:                            OLS   Adj. R-squared:                  0.213
Method:                 Least Squares   F-statistic:                     46.63
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           1.35e-26
Time:                        01:12:01   Log-Likelihood:                -1744.2
No. Observations:                 506   AIC:                             3496.
Df Residuals:                     502   BIC:                             3513.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept                1.2010      2.029      0.592      0.554      -2.785       5.187
Boston[t]               -0.4491      0.465     -0.966      0.335      -1.362       0.464
I(pow(Boston[t], 2))     0.0558      0.030      1.852      0.065      -0.003       0.115
I(pow(Boston[t], 3))    -0.0009      0.001     -1.517      0.130      -0.002       0.000
==============================================================================
Omnibus:                      607.734   Durbin-Watson:                   1.239
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            53621.219
Skew:                           5.726   Prob(JB):                         0.00
Kurtosis:                      52.114   Cond. No.                     5.20e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.2e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   crim   R-squared:                       0.420
Model:                            OLS   Adj. R-squared:                  0.417
Method:                 Least Squares   F-statistic:                     121.3
Date:                Sat, 05 Jan 2019   Prob (F-statistic):           4.45e-59
Time:                        01:12:01   Log-Likelihood:                -1668.5
No. Observations:                 506   AIC:                             3345.
Df Residuals:                     502   BIC:                             3362.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               53.1655      3.356     15.840      0.000      46.571      59.760
Boston[t]               -5.0948      0.434    -11.744      0.000      -5.947      -4.242
I(pow(Boston[t], 2))     0.1555      0.017      9.046      0.000       0.122       0.189
I(pow(Boston[t], 3))    -0.0015      0.000     -7.312      0.000      -0.002      -0.001
==============================================================================
Omnibus:                      569.730   Durbin-Watson:                   1.359
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            47929.717
Skew:                           5.106   Prob(JB):                         0.00
Kurtosis:                      49.573   Cond. No.                     3.67e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.67e+05. This might indicate that there are
strong multicollinearity or other numerical problems.

==============================================================================

