# dataset-testing

This repository contains different regression models implemented in Python using scikit-learn. The project focuses on Ontario healthcare data, building and evaluating two different regression models to analyze and predict outcomes.


Install the required libraries using: pip install pandas numpy scikit-learn matplotlib seaborn
Run the script in the terminal by typing python main.py or python3 main.py (dependent on your version of python)

What the program actually does:
- loads healthcare data (.csv file), cleans it, uses imputation methods in order to account for missing values, and overall prepares it for the actual regression
- then different regression methods are utilized in order to classify related health-care outcomes.
- generates a confusion matrix and calculates the precision, recall, f1-score, and accuracy to assess the model's performance
- and finally, since different regression models are utilized alongside various formats of datasets, we can effective compare the different regression techniques (includes random forest, linear regression, polynomial regression, and multinomial regression)


After the program is executed, the program will output:
- confusion matrix that represents the model's classification performance
- precision, recall, f1-scores, and accuracy metrics within a data table to evaluate how well the model actually perfomed with the given dataset (letsyou know whether better data is required)


Different regression models:
- Random Forest Regression: this is a learned method that constrcusts multiple decision trees during the training period and averages the different predictions in order to improve accuaracy and reduce any overfitting --> good for non-linear datasets and can help analyze a variety of different results

- Linear Regression: models the relationship between an independent variable (X) and a dependent variable (Y) by fitting a straight line (Y = mX + b).

- Polynomial Regression: extends linear regression by introducing polynomial terms (Y=aX^2+bX+c). It can capture non-linear relationships while remaining a form of linear regression.

- Multinomial Regression: is an extension of logistic regression for predicting multiple categorical outcomes (rather than just binary classification). It uses the softmax function to assign probabilities to multiple classes.
