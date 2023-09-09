# ISLP - Introduction to Statistical learning Python
# Wage dataset EDA

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# read Wage dataset
df = pd.read_csv('../Data/Wage.csv', index_col='SlNo')
df

df.info()

# verify missing values
df.isnull().sum().sum()

# drop constant value columns
def drop_constant_column(df):
    """
    Drops constant value columns of pandas dataframe (df).
    """
    return df.loc[:, (df != df.iloc[0]).any()]

df = drop_constant_column(df)

# descriptive statistics
df.describe().T

# plot histograms
num_columns = df.select_dtypes(include = np.number).columns

for i in range(0, len(num_columns)):
  fig = plt.figure(figsize = (10, 5))
  sns.histplot(x = df[num_columns[i]])

# plot boxplots
for i in range(0, len(num_columns)):
  fig = plt.figure(figsize = (10, 5))
  sns.boxplot(x = df[num_columns[i]])

# plot bar charts
cat_columns = df.select_dtypes(include = 'object').columns

for i in range(0, len(cat_columns)):
  fig = plt.figure(figsize = (10, 5))
  sns.countplot(x = df[cat_columns[i]])

# clean categorical columns (remove '1. ', '2. ', ...)
for i in range(0, len(cat_columns)):
  df[cat_columns[i]] = df[cat_columns[i]].str.replace(r'^[1-5]\. ', '', regex=True)

# pairplots
# create `marital_status` variable
df['marital_status'] = np.where(df['maritl'] == 'Married', 1, 0)

# pairplot by `marital_status`
sns.pairplot(df, hue = 'marital_status', kind='reg', palette='Set1')

# pairplot by `health_ins`
sns.pairplot(df, hue = 'health_ins', kind='reg', palette='Set1')

# `wage` by `age` regression plot
fig = plt.figure(figsize = (10, 8))
sns.regplot(data = df, x = 'age', y = 'wage', lowess = True, 
            scatter_kws={"color": "gray", 'alpha': 0.1}, 
            line_kws={"color": "blue"})
plt.title('Income Survey from Males Central Atlantic Region - USA in 2009')
plt.show()

# `logwage` by `age` regression plot
fig = plt.figure(figsize = (10, 8))
sns.regplot(data = df, x = 'age', y = 'logwage', lowess = True, 
            scatter_kws={"color": "gray", 'alpha': 0.1}, 
            line_kws={"color": "blue"})
plt.title('Income Survey from Males Central Atlantic Region - USA in 2009')
plt.show()

# `wage` by `year` regression plot
fig = plt.figure(figsize = (10, 8))
sns.regplot(data = df, x = 'year', y = 'wage', lowess = True, 
            scatter_kws={"color": "gray", 'alpha': 0.2}, 
            line_kws={"color": "blue"})
plt.title('Income Survey from Males Central Atlantic Region - USA in 2009')
plt.show()

# `wage` by `education` boxplots
fig = plt.figure(figsize = (10, 8))
group_medians =df.groupby(['education'])['wage'].median().sort_values(ascending=True)
sns.boxplot(data = df, x = 'education', y = 'wage', 
            order = group_medians.index)

# combine the three plots
fig, axs = plt.subplots(ncols=3, figsize=(12, 8))
sns.regplot(data = df, x = 'age', y = 'wage', lowess = True, 
            scatter_kws={"color": "gray", 'alpha': 0.1}, 
            line_kws={"color": "blue"}, 
            ax = axs[0])
sns.regplot(data = df, x = 'year', y = 'wage', lowess = True, 
            scatter_kws={"color": "gray", 'alpha': 0.2}, 
            line_kws={"color": "blue"}, 
            ax = axs[1])
group_medians =df.groupby(['education'])['wage'].median().sort_values(ascending=True)
sns.boxplot(data = df, x = 'education', y = 'wage', order = group_medians.index, 
            ax = axs[2])
plt.xticks(rotation = 60)
