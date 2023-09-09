# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
# import libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
# from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# import data visualisation tools
import matplotlib.pyplot as plt
import xkcd
%matplotlib inline
# from matplotlib import pylab
# import plotly.plotly as py
# import plotly.graph_objs as go
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 10)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
url = "../Data/Auto.csv"
Auto = pd.read_csv(url)
#
#
#
#
#
Auto.head()
#
#
#
#
#
Auto.info()
#
#
#
#
#
Auto.isnull().sum().sum()
#
#
#
#
#
#
#
Auto.describe().T
#
#
#
#
#
Auto.hist()
#
#
#
#
#
y = Auto.mpg.astype(float)
x = Auto.horsepower.astype(float)
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
#
#
#
#
#
model.summary()
#
#
#
#
#
#
#
#
#
#
#
#
#
model.resid.std(ddof=X.shape[1])
#
#
#
#
#
#
#
values = slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f'Slope: {values[0]:.4f}')
print(f'Intercept (constant): {values[1]:.4f}')
print(f'R-value (Pearson coefficient): {values[2]:.4f}')
print(f'R-squared (coefficient of determination): {values[2]**2:.4f}')
print(f'p-value: {values[3]:.4f}')
#
#
#
#
#
plt.figure(figsize=(25, 10))
plotdata = pd.concat([x, y], axis = 1)
sns.lmplot(x = "horsepower", y = "mpg", data = plotdata)
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()
#
#
#
#
#
#
#
# confidence interval/prediction confidence interval (statsmodels)
new_data = np.array([1, 98])
pred = model.get_prediction(new_data)
pred.summary_frame(alpha = 0.05)
#
#
#
#
#
#
#
#
#
#
#
sns.histplot(model.resid)
#
#
#
#
#
mu, std = stats.norm.fit(model.resid)
print(mu, std)
#
#
#
#
stats.shapiro(model.resid)
#
#
#
#
#
#
#
from statsmodels.nonparametric.smoothers_lowess import lowess

# function to plot residuals vs. fitted values
def resid_fitted_plot(model):

    residuals = model.resid
    fitted = model.fittedvalues
    smoothed = lowess(residuals, fitted)
    top3 = abs(residuals).sort_values(ascending = False)[:3]

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
    ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Fitted Values')
    ax.set_title('Residuals vs. Fitted')
    ax.plot([min(fitted), max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)

    for i in top3.index:
        ax.annotate(i, xy=(fitted[i],residuals[i]))

    plt.show()

resid_fitted_plot(model = model)
#
#
#
#
#
#
#
sm.qqplot(model.resid, line='s')
plt.show()
#
#
#
#
#
#
#
# function to plot standardized residuals vs. fitted values
def std_resid_fitted_plot(model):

    student_residuals = model.get_influence().resid_studentized_internal
    fitted = model.fittedvalues
    sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
    sqrt_student_residuals.index = model.resid.index
    smoothed = lowess(sqrt_student_residuals, fitted)
    top3 = abs(sqrt_student_residuals).sort_values(ascending = False)[:3]

    fig, ax = plt.subplots()
    ax.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
    ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
    ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
    ax.set_xlabel('Fitted Values')
    ax.set_title('Scale-Location')
    ax.set_ylim(0,max(sqrt_student_residuals)+0.1)
    for i in top3.index:
        ax.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
    plt.show()

std_resid_fitted_plot(model = model)
#
#
#
#
#
#
#
#
#
#
#
sns.pairplot(Auto, palette='Dark2')
#
#
#
#
sns.pairplot(Auto, hue = 'origin', palette='Dark2')
#
#
#
#
#
Auto.corr()
#
#
#
#
#
sns.heatmap(Auto.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
#
#
#
#
#
# X = Auto[['cylinders', 'displacement', 'horsepower', 'weight',
#        'acceleration', 'year', 'origin']]
# Y = Auto['mpg']
# X1 = sm.add_constant(X)
# reg = sm.OLS(Y, X1).fit()
#
#
#
# `cylinders` and `origin` are categorical variables
reg = ols('mpg ~ C(cylinders) + displacement + horsepower + weight + acceleration + year + C(origin)', data = Auto).fit()
#
#
#
#
#
reg.summary()
#
#
#
#
#
#
#
#
#
#
#
sm.stats.anova_lm(reg, typ = 2)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
resid_fitted_plot(model = reg)
#
#
#
#
#
#
#
sm.qqplot(reg.resid, line='s')
plt.show()
#
#
#
#
#
#
#
std_resid_fitted_plot(model = reg)
#
#
#
#
#
#
#
#
#
reg_1 = ols('mpg ~ C(cylinders) + displacement + horsepower + weight + acceleration + year + C(origin) + C(cylinders) * displacement', data = Auto).fit()
#
#
#
#
reg_1.summary()
#
#
#
#
#
#
#
reg_2 = ols('mpg ~ C(cylinders) + displacement + horsepower + weight + acceleration + year + C(origin) + weight * displacement', data = Auto).fit()
#
#
#
#
reg_2.summary()
#
#
#
#
#
#
#
#
#
#
reg_3 = ols('mpg ~ C(cylinders) + displacement + horsepower + weight + acceleration + year + C(origin) + np.log(horsepower) + np.sqrt(acceleration)', data = Auto).fit()
#
#
#
#
reg_3.summary()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
file = "../Data/Carseats.csv"
CarSeats = pd.read_csv(file)
#
#
#
#
#
CarSeats.head()
#
#
#
#
#
CarSeats.info()
#
#
#
#
#
CarSeats.isnull().sum().sum()
#
#
#
#
#
#
#
CarSeats.describe().T
#
#
#
#
#
CarSeats.hist()
#
#
#
#
#
reg = ols(formula = 'Sales ~ Price + C(Urban) + C(US)', data = CarSeats).fit() # C prepares categorical data for regression
#
#
#
#
#
reg.summary()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
reg_1 = ols(formula = 'Sales ~ Price + C(US)', data = CarSeats).fit()
#
#
#
#
#
reg_1.summary()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
resid_fitted_plot(model = reg_1)
#
#
#
#
#
#
#
fig = plt.figure(figsize = (25, 15))
fig.set_size_inches(30, fig.get_figheight(), forward=True)
sm.graphics.influence_plot(reg_1, criterion="cooks", size = 0.0002**2)
plt.title("Residuals vs. Leverage")
fig = plt.gcf()
fig.set_size_inches(25, 15)
plt.show()
#
#
#
#
#
