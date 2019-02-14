# ISLP

## What is ISLP?

[An Introduction to Statistical Learning with Applications in R](http://www-bcf.usc.edu/~gareth/ISL/) ("ISLR" for short) is a great practical introduction to machine learning. This repository is my personal attempt to "translate" the R Code in the book (contained in R Labs and Applied Exercises as well as some Conceptual Exercises) into Python. I have fancifully replaced the R in ISLR with P to reflect the change in programming language.

The repository is built into different sections within the R Lab and Applied Exercises. I am not providing solutions to any Conceptual Exercises, but will provide the Python code for any answers in that section, if required. I will include any relevant concept or rationale as comments within the code.  

## How is this repository arranged?

This repository is divided into chapters. Each chapter is divided into two subfolders - 
1. **R Lab:** This contains Python Codes from the R Labs section. These are marked according to the relevant sections. Therefore, a file named 3.6.3.ipynb means it contains code in Jupyter Notebook from Section 3.6.3 of the R Lab (within Chapter 3).
2. **Applied Exercises:** This contains Python Codes from Applied Exercises from each chapter. The Applied Exercises will contain some sub-questions which ask the reader to interpret the model rather than code it. I will provide solutions to those questions for the sake of completeness.

## Table of Contents
1. [Chapter 3: Linear Regression](https://bit.ly/2VsS4cL)
2. [Chapter 4: Classification](http://bit.ly/2H862gG)
3. [Chapter 5: Resampling Methods](http://bit.ly/2RIQ4Ou)
4. [Chapter 6: Linear Model Selection and Regularization](http://bit.ly/2FEiza8)
5. [Chapter 7: Moving Beyod Linearity](http://bit.ly/2CNtKK7)
6. [Chapter 8: Tree-Based Methods](http://bit.ly/2sTm6Jm)
7. [Chapter 9: Support Vector Machines](http://bit.ly/2sXz1Ko)
8. [Chapter 10: Unsupervised Learning](http://bit.ly/2GlxkiJ)
9. [Extra: Data](http://bit.ly/2MmkroK)

## Notes
1. I have used Python 3.6.6 for the purposes of programmming all codes in this book. You might want to check the Python version in your computer before applying my codes since syntaxes might differ. This is particularly applicable for those running Python 2.x.
2. While reading a file in Python using `pd.read_csv(url)`, I use the location of files *in my computer* as the URL. You will need to use the location of files *in your computer* while reading those files in Python.
3. The book uses many data sets that are inbuilt in the ISLR library. I have uploaded them in the [Data](http://bit.ly/2MmkroK) folder.
4. Since all codes in this repository have been typed in Jupyter Notebook, there will be a *slight* difference between the codes in the repository and if you use those codes in an IDE/Terminal. This difference will reflect during printing a result. As an example, what I type in as `df.head()` in this repository should be typed as `print(df.head())` in an IDE/Terminal.
5. I ignore warnings using the `warnings.filterwarnings('ignore')` command for presentational aesthetics. That does not mean I ignore said warnings. I incorporate warnings and implement them (e.g. using `solver=` while performing logistic regression through `sklearn.linear_model.LogisticRegression` or using `test_size=` instead of `train_size=` while splitting the data set using `sklearn.model_selection.train_test_split()`) before igoring them.
6. The codes start at [Chapter 3](https://bit.ly/2VsS4cL) because Chapter 1 contains no code and Chapter 2 is a basic introduction to coding in R, which has simple translations in Python. However, in my opinion, Chapter 2 is perhaps the most important chapter in the book since it explains the most important concepts of statistical learning (such as bias-variance tradeoff) which apply to all of the book irrespective of the model you choose to apply.
7. I have not performed Best Subset Selection in [Question 11.a.](https://github.com/arpanganguli/ISLP/blob/master/Chapter%206/Applied%20Exercises/11.ipynb) of Applied Exercises in Chapter 6. This is because the said method is extremely time-consuming and I have provided the code for best subset selection in previous examples and exercises. This means that my answer for Question 11 overall will not include results from best subset selection. That is not say that I will never solve it. I will return to it at a later date when I am little less occupied.
8. I have omitted R Lab 6.5.3: Choosing Among Models Using the Validation Set Approach and Cross-Validation. I have covered that code through the previous two R Labs, [6.5.1: Best Subset Selection](https://github.com/arpanganguli/ISLP/blob/master/Chapter%206/R%20Lab/6.5.1%20Best%20Subset%20Selection.ipynb) and [6.5.2: Forward and Backward Stepwise Selection](https://github.com/arpanganguli/ISLP/blob/master/Chapter%206/R%20Lab/6.5.2%20Forward%20and%20Backward%20Stepwise%20Selection.ipynb). The same applies to R Lab 10.6: NCI60 Data Example which is an application of the previous two R Labs, [10.5.1: Principal Components Analysis](https://github.com/arpanganguli/ISLP/blob/master/Chapter%2010/R%20Lab/10.4%20Lab%201%20%20-%20Principal%20Components%20Analysis.ipynb) and [10.5.2: Clustering](https://github.com/arpanganguli/ISLP/blob/master/Chapter%2010/R%20Lab/10.5%20Lab%202%20-%20Clustering.ipynb) to a particular data set([NCI60](https://github.com/arpanganguli/ISLP/blob/master/Data/NCI60.csv)).
9. I have not solved Question 10 in [Applied Exercises](http://bit.ly/2BzonOw) of Chapter 6. This is because my computer is constantly hanging when performing best subset selection and I have wasted close to 4 hours in this pursuit. I will come back to this question at a later date.
10. Python falls woefully short when it comes to having packages required to model Generalized Additive Models (GAMs). My personal recommendation will be to use R to model GAMs. This is applicable for both Section [7.8.3: GAMs](https://github.com/arpanganguli/ISLP/blob/master/Chapter%207/R%20Lab/7.8.3%20GAMs.ipynb) in R Labs and [Question 10](https://github.com/arpanganguli/ISLP/blob/master/Chapter%207/Applied%20Exercises/10.ipynb) in Applied Exercises in Chapter 7.
