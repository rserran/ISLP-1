# ISLP

## What is ISLP?

[An Introduction to Statistical Learning with Applications in R](http://www-bcf.usc.edu/~gareth/ISL/) ("ISLR" for short) is a great practical introduction to machine learning. This repository is my personal attempt to "translate" as much code as I can into Python as I go along. I have fancifully replaced the R in ISLR with P to reflect the change in language.

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
5. Chapter 7: Moving Beyod Linearity
6. Chapter 8: Tree-Based Methods
7. Chapter 9: Support Vector Machines
8. Chapter 10: Unsupervised Learning
9. [Extra: Data](http://bit.ly/2MmkroK)

## Notes
1. The codes start at Chapter 3 because the Chapter 1 contains no code and the Chapter 2 is a basic introduction to coding in R, which has simple translations in Python. However, in my opinion, Chapter 2 is perhaps the most important chapter in the book since it explains the most important concepts of statistical learning (such as bias-variance tradeoff) which apply to all of the book irrespective of the model you choose to apply.
2. I have not performed Best Subset Selection in Question 11.a. of Applied Exercises in Chapter 6. This is because the said method is extremely time-consuming and I have provided the code for best subset selection in previous examples and exercises. This means that my answer for Question 11 overall will not include results from best subset selection. That is not say that I will never solve it. I will return to it at a later date when I am little less occupied.
3. I have omitted R Lab 6.5.3: "Choosing Among Models Using the Validation Set Approach and Cross-Validation". I have covered that code through the previous two R Labs, 6.5.1 and 6.5.2.
4. I have not solved Question 10 in Chapter 6. This is because my computer is constantly hanging when performing best subset selection and I have wasted close to 4 hours in this pursuit. I will come back to this question later.
