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

# 13.a. create standard normal vector
x = np.random.standard_normal(100)
plt.figure(figsize = (11, 5))
sns.distplot(x, color = 'g')
plt.title("x ~ N(0,1)")

# 13.b. create eps ~ N(0,0.25)
eps = np.random.normal(0, 0.25, 100)
plt.figure(figsize = (11, 5))
sns.distplot(eps, color = 'y')
plt.title("eps ~ N(0,0.25)")

# 13.c. generate vector y = -1 + 0.5*x + eps

y = -1 + (0.5*x) + eps
print("Length of vector y: ", len(y))
print("Beta_0: ", -1.0)
print("Beta_1: ", 0.5)

# 13.d. generate scatterplot
plt.figure(figsize = (11, 5))
plt.scatter(y, x)
plt.title("xy-scatterplot")
plt.xlabel("x")
plt.ylabel("y")

"""
Observation: linear relationship between x and y with a positive slope
"""

# 13.e. fitting a linear model
data = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis = 1)
data.columns = ['x', 'y']
reg_1 = ols("y~x", data = data).fit()
f1 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f1)
print(reg_1.summary())
print()
print("Beta_hat_0: %f" % reg_1.params[0])
print("Beta_hat_1: %f" % reg_1.params[1])
print()
print("Difference in Beta_0: ", -1.0 - reg_1.params[0])
print("Difference in Beta_1: ", 0.5 - reg_1.params[1])
print()

# 13.f. displaying the least squares line calculated in 13.d.
plt.figure(figsize = (11, 5))
reg_plot = sns.regplot(x, y, data = data)
regline = reg_plot.get_lines()[0]
regline.set_color('green')
plt.title("xy-regression_plot")
plt.xlabel("x")
plt.ylabel("y")
labels = ['x', 'y']
plt.legend(labels)

# 13.g. polynomial regression
reg_2 = ols("y~x+I(pow(x, 2))", data = data).fit()
f2 = np.poly1d(np.polyfit(x, y, 2))
print("y = ", f2)
print(reg_2.summary())
print()
RMSE_1 = np.sqrt(reg_1.mse_model) # root mean squared error of the first regression model
RMSE_2 = np.sqrt(reg_2.mse_model) # root mean squared error of the first regression model
print("RMSE_1:", RMSE_1) # this value in the range ~ 5.0
print("RMSE_2:", RMSE_2) # this value in the range of ~ 3.5
print()
print(colored('='*78, 'green'))
print()

"""
There is not much to choose between the two models given their R^2. Given that
RMSE_2 is lower than RMSE_1 in general, this suggests the polynomial
model fits the data better. This is because it is able to fit the non-linear
nature of the true model better.
"""

# 13.h. regression after reducing variance
eps_r = eps/4 # the _r connotes "reduced"
y_r = -1 + (0.5*x) + eps_r

print("Variance of eps: ", eps.var())
print("Variance of eps_r: ", eps_r.var()) # confirms that the overall variance is reduced

plt.figure(figsize = (11, 5))
plt.scatter(y_r, x)
plt.title("xy_r-scatterplot")
plt.xlabel("x")
plt.ylabel("y_r")

data_r = pd.concat([pd.DataFrame(x), pd.DataFrame(y_r)], axis = 1)
data.columns = ['x', 'y_r']
reg_r_1 = ols("y_r~x", data = data).fit()
f_r_1 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_r_1)
print(reg_r_1.summary())
print()
print("Beta_hat_r_0: %f" % reg_r_1.params[0])
print("Beta_hat_r_1: %f" % reg_r_1.params[1])
print()
print("Difference in Beta_0: ", -1.0 - reg_r_1.params[0])
print("Difference in Beta_1: ", 0.5 - reg_r_1.params[1])
print()

plt.figure(figsize = (11, 5))
reg_plot_r = sns.regplot(x, y_r, data = data_r)
regline_r = reg_plot_r.get_lines()[0]
regline_r.set_color('green')
plt.title("xy_r-regression_plot")
plt.xlabel("x")
plt.ylabel("y_r")
labels_r = ['x', 'y_r']
plt.legend(labels_r)

reg_r_2 = ols("y_r~x+I(pow(x, 2))", data = data_r).fit()
f_r_2 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_r_2)
print(reg_r_2.summary())
print()
RMSE_r_1 = np.sqrt(reg_r_1.mse_model) # root mean squared error of the first regression model
RMSE_r_2 = np.sqrt(reg_r_2.mse_model) # root mean squared error of the first regression model
print("RMSE_r_1:", RMSE_r_1) # this value in the range ~ 5.0
print("RMSE_r_2:", RMSE_r_2) # this value in the range of ~ 3.5
print()
print(colored('='*78, 'green'))
print()

"""
Given that RMSE_r_2 is lower than RMSE_r_1 in general, this suggests the
polynomial model fits the data better. The R_2 of the polynomial model also
significantly larger than the first model. This suggests the polynomial model
better explains the reduced variation in data without compromising on the fit.
"""
# 13.i. regression after increasing the variance
eps_i = eps*4 # the _i connotes "increased"
y_i = -1 + (0.5*x) + eps_i

print("Sum of square of eps: ", eps.var())
print("Sum of square of eps_i: ",eps_i.var()) # confirms that the overall variance is reduced

plt.figure(figsize = (11, 5))
plt.scatter(y_i, x)
plt.title("xy_i-scatterplot")
plt.xlabel("x")
plt.ylabel("y_i")

data_i = pd.concat([pd.DataFrame(x), pd.DataFrame(y_i)], axis = 1)
data.columns = ['x', 'y_i']
reg_i_1 = ols("y_i~x", data = data).fit()
f_i_1 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_i_1)
print(reg_i_1.summary())
print()
print("Beta_hat_i_0: %f" % reg_i_1.params[0])
print("Beta_hata_i_1: %f" % reg_i_1.params[1])
print()
print("Difference in Beta_0: ", -1.0 - reg_i_1.params[0])
print("Difference in Beta_1: ", 0.5 - reg_i_1.params[1])
print()

plt.figure(figsize = (11, 5))
reg_plot_i = sns.regplot(x, y_i, data = data_i)
regline_i = reg_plot_i.get_lines()[0]
regline_i.set_color('green')
plt.title("xy_i-regression_plot")
plt.xlabel("x")
plt.ylabel("y_i")
labels_i = ['x', 'y_i']
plt.legend(labels_i)

reg_i_2 = ols("y_i~x+I(pow(x, 2))", data = data_i).fit()
f_i_2 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_i_2)
print(reg_i_2.summary())
print()
RMSE_i_1 = np.sqrt(reg_i_1.mse_model) # root mean squared error of the first regression model
RMSE_i_2 = np.sqrt(reg_i_2.mse_model) # root mean squared error of the first regression model
print("RMSE_i_1:", RMSE_i_1) # this value in the range ~ 5.0
print("RMSE_i_2:", RMSE_i_2) # this value in the range of ~ 3.5
print()
print(colored('='*78, 'green'))
print()

"""
Given that RMSE_r_2 is lower than RMSE_r_1 in general, this suggests the
polynomial model fits the data better. However, the R^2 of the second model is
significantly lower than the first model, which suggests that the polynomial
model starts to follow the noise thereby showing increased variance.
"""

# 13.j. confidence intervals

confint1 = pd.DataFrame(reg_1.conf_int(alpha = 0.05)).T
confint2 = pd.DataFrame(reg_2.conf_int(alpha = 0.05)).T
confint_r_1 = pd.DataFrame(reg_r_1.conf_int(alpha = 0.05)).T
confint_r_2 = pd.DataFrame(reg_r_2.conf_int(alpha = 0.05)).T
confint_i_1 = pd.DataFrame(reg_i_1.conf_int(alpha = 0.05)).T
confint_i_2 = pd.DataFrame(reg_i_2.conf_int(alpha = 0.05)).T

print("95% C.I. of Linear Model:\n", confint1)
print("95% C.I. of Polynomial Model:\n", confint2)
print()
print("95% C.I. of Linear Model with reduced variance:\n", confint_r_1)
print("95% C.I. of Polynomial Model with reduced variance:\n", confint_r_2)
print()
print("95% C.I. of Linear Model with increased variance:\n", confint_r_1)
print("95% C.I. of Polynomial Model with increased variance:\n", confint_r_2)
print()
print(colored('='*78, 'green'))
print()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESULTS (NOT PART OF CODE)
runfile('/Users/arpanganguli/Documents/Finance/ISLR/General_Code.py', wdir='/Users/arpanganguli/Documents/Finance/ISLR')
Length of vector y:  100
Beta_0:  -1.0
Beta_1:  0.5
y =   
0.5469 x - 0.9989
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.815
Model:                            OLS   Adj. R-squared:                  0.813
Method:                 Least Squares   F-statistic:                     432.0
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           1.07e-37
Time:                        20:31:40   Log-Likelihood:                 3.0314
No. Observations:                 100   AIC:                            -2.063
Df Residuals:                      98   BIC:                             3.148
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.9989      0.024    -42.106      0.000      -1.046      -0.952
x              0.5469      0.026     20.784      0.000       0.495       0.599
==============================================================================
Omnibus:                        1.588   Durbin-Watson:                   2.112
Prob(Omnibus):                  0.452   Jarque-Bera (JB):                1.086
Skew:                           0.047   Prob(JB):                        0.581
Kurtosis:                       3.502   Cond. No.                         1.11
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Beta_hat_0: -0.998911
Beta_hat_1: 0.546912

Difference in Beta_0:  -0.0010894680611319707
Difference in Beta_1:  -0.046911604188337064

y =            2
-0.02544 x + 0.5548 x - 0.978
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.818
Model:                            OLS   Adj. R-squared:                  0.814
Method:                 Least Squares   F-statistic:                     217.6
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           1.39e-36
Time:                        20:31:40   Log-Likelihood:                 3.7556
No. Observations:                 100   AIC:                            -1.511
Df Residuals:                      97   BIC:                             6.304
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.9780      0.029    -33.178      0.000      -1.037      -0.920
x                0.5548      0.027     20.484      0.000       0.501       0.609
I(pow(x, 2))    -0.0254      0.021     -1.190      0.237      -0.068       0.017
==============================================================================
Omnibus:                        1.511   Durbin-Watson:                   2.165
Prob(Omnibus):                  0.470   Jarque-Bera (JB):                0.969
Skew:                           0.138   Prob(JB):                        0.616
Kurtosis:                       3.396   Cond. No.                         2.26
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

RMSE_1: 4.9285307332738695
RMSE_2: 3.4906775267674326

==============================================================================

Variance of eps:  0.05689272407334686
Variance of eps_r:  0.003555795254584179
y =   
0.5469 x - 0.9989
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    y_r   R-squared:                       0.815
Model:                            OLS   Adj. R-squared:                  0.813
Method:                 Least Squares   F-statistic:                     432.0
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           1.07e-37
Time:                        20:31:40   Log-Likelihood:                 3.0314
No. Observations:                 100   AIC:                            -2.063
Df Residuals:                      98   BIC:                             3.148
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.9989      0.024    -42.106      0.000      -1.046      -0.952
x              0.5469      0.026     20.784      0.000       0.495       0.599
==============================================================================
Omnibus:                        1.588   Durbin-Watson:                   2.112
Prob(Omnibus):                  0.452   Jarque-Bera (JB):                1.086
Skew:                           0.047   Prob(JB):                        0.581
Kurtosis:                       3.502   Cond. No.                         1.11
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Beta_hat_r_0: -0.998911
Beta_hat_r_1: 0.546912

Difference in Beta_0:  -0.0010894680611319707
Difference in Beta_1:  -0.046911604188337064

y =   
0.5469 x - 0.9989
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    y_r   R-squared:                       0.984
Model:                            OLS   Adj. R-squared:                  0.984
Method:                 Least Squares   F-statistic:                     3039.
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           3.25e-88
Time:                        20:31:40   Log-Likelihood:                 142.39
No. Observations:                 100   AIC:                            -278.8
Df Residuals:                      97   BIC:                            -271.0
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.9945      0.007   -134.951      0.000      -1.009      -0.980
x                0.5137      0.007     75.863      0.000       0.500       0.527
I(pow(x, 2))    -0.0064      0.005     -1.190      0.237      -0.017       0.004
==============================================================================
Omnibus:                        1.511   Durbin-Watson:                   2.165
Prob(Omnibus):                  0.470   Jarque-Bera (JB):                0.969
Skew:                           0.138   Prob(JB):                        0.616
Kurtosis:                       3.396   Cond. No.                         2.26
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

RMSE_r_1: 4.9285307332738695
RMSE_r_2: 3.2611817048240397

==============================================================================

Sum of square of eps:  0.05689272407334686
Sum of square of eps_i:  0.9102835851735498
y =   
0.5469 x - 0.9989
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    y_i   R-squared:                       0.815
Model:                            OLS   Adj. R-squared:                  0.813
Method:                 Least Squares   F-statistic:                     432.0
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           1.07e-37
Time:                        20:31:40   Log-Likelihood:                 3.0314
No. Observations:                 100   AIC:                            -2.063
Df Residuals:                      98   BIC:                             3.148
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.9989      0.024    -42.106      0.000      -1.046      -0.952
x              0.5469      0.026     20.784      0.000       0.495       0.599
==============================================================================
Omnibus:                        1.588   Durbin-Watson:                   2.112
Prob(Omnibus):                  0.452   Jarque-Bera (JB):                1.086
Skew:                           0.047   Prob(JB):                        0.581
Kurtosis:                       3.502   Cond. No.                         1.11
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Beta_hat_i_0: -0.998911
Beta_hata_i_1: 0.546912

Difference in Beta_0:  -0.0010894680611319707
Difference in Beta_1:  -0.046911604188337064

y =   
0.5469 x - 0.9989
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    y_i   R-squared:                       0.313
Model:                            OLS   Adj. R-squared:                  0.299
Method:                 Least Squares   F-statistic:                     22.14
Date:                Fri, 04 Jan 2019   Prob (F-statistic):           1.20e-08
Time:                        20:31:40   Log-Likelihood:                -134.87
No. Observations:                 100   AIC:                             275.7
Df Residuals:                      97   BIC:                             283.6
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.9121      0.118     -7.735      0.000      -1.146      -0.678
x                0.7193      0.108      6.639      0.000       0.504       0.934
I(pow(x, 2))    -0.1018      0.086     -1.190      0.237      -0.272       0.068
==============================================================================
Omnibus:                        1.511   Durbin-Watson:                   2.165
Prob(Omnibus):                  0.470   Jarque-Bera (JB):                0.969
Skew:                           0.138   Prob(JB):                        0.616
Kurtosis:                       3.396   Cond. No.                         2.26
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

RMSE_i_1: 4.9285307332738695
RMSE_i_2: 4.453531569897252

==============================================================================

95% C.I. of Linear Model:
    Intercept         x
0  -1.045989  0.494693
1  -0.951832  0.599131
95% C.I. of Polynomial Model:
    Intercept         x  I(pow(x, 2))
0  -1.036521  0.501058     -0.067889
1  -0.919511  0.608574      0.017004

95% C.I. of Linear Model with reduced variance:
    Intercept         x
0  -1.045989  0.494693
1  -0.951832  0.599131
95% C.I. of Polynomial Model with reduced variance:
    Intercept         x  I(pow(x, 2))
0  -1.009130  0.500265     -0.016972
1  -0.979878  0.527144      0.004251

95% C.I. of Linear Model with increased variance:
    Intercept         x
0  -1.045989  0.494693
1  -0.951832  0.599131
95% C.I. of Polynomial Model with increased variance:
    Intercept         x  I(pow(x, 2))
0  -1.009130  0.500265     -0.016972
1  -0.979878  0.527144      0.004251
