# ISLP Chapter 5 - Resampling
# Source: https://dadataguy.medium.com/logistic-regression-using-statsmodels-a63e7944de76

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from patsy import dmatrices

#SNS Settings 
sns.set(color_codes = True)
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(10,10)})
sns.set_palette("Set3")

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# read dataset
url = "../../Data/Default.csv"
default = pd.read_csv(url, index_col = 'Unnamed: 0')

default.head()

default.info()

# count 'default'
default['default'].value_counts()

# convert 'default' to binary
default['default'] = (default['default'] == 'Yes').astype(int)

# fit a logistic regression model with 'income' and 'balance' to predict 'default'
y, X = dmatrices('default ~ income + balance', data = default, 
                return_type = 'dataframe')

# logistic regression model
model_1 = sm.Logit(y, X)

res_1 = model_1.fit()

res_1.summary()

# generate predictions
y_pred = (res_1.predict(X) > 0.5).astype(int)
y_pred

# accuracy and confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y['default'].values, y_pred.values)
print ("Confusion Matrix : \n", cm)

print('Test accuracy = ', accuracy_score(y['default'].values, y_pred.values))

# create anew data frame with p_values
p_values = res_1.pvalues[1:].T.to_frame()

p_values.index.name = 'Features'
p_values.reset_index()

# creating new data frame using the coefficients
params =  res_1.params[1:].T.to_frame()

params.index.name = 'Features'
params.reset_index()

# merge params and pvalues, fill NaN values with 0
results = pd.merge(params, p_values, how = "left", on = "Features", 
                   suffixes=("params","pvalues")).fillna(0).reset_index()

# rename columns and display results
results = results.rename(columns={'0params': 'Params', "0pvalues":'Pvalues'})
results

# generate the final data frame with the filters
final = results.loc[(results['Pvalues'] < 0.05) & (results['Params'] >= 0.00)] \
    .reset_index(drop=True)

final

# Creating a new column called Odds
final['Odds'] = np.exp(final['Params'])

# Creating a new column called percent using the log odds of the feature
final['Percent'] = (final['Odds'] - 1)*100

final.sort_values(by=['Odds'], ascending=False).reset_index(drop=True)

# Does the addition of 'student' improves the model?

# convert 'student' to binary
default['student'] = (default['student'] == 'Yes').astype(int)

# fit a logistic regression model adding 'student' to model_1
y, X = dmatrices('default ~ income + balance + student', data = default, 
                return_type = 'dataframe')

# logistic regression model
model_2 = sm.Logit(y, X)

res_2 = model_2.fit()

res_2.summary()

# generate predictions
y_pred = (res_2.predict(X) > 0.5).astype(int)
y_pred

# accuracy and confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y['default'].values, y_pred.values)
print ("Confusion Matrix : \n", cm)

print('Test accuracy = ', accuracy_score(y['default'].values, y_pred.values))

# Comment: Adding 'student' variable did not improve the model.