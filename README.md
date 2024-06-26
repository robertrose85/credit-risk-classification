# credit-risk-classification

## Overview of the Analysis

Our goal for this analysis is to build a model that will identify whether a particular borrower is creditworthy. To build this model we are using a dataset of historical lending activity from a peer-to-peer lending services company.

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

Dataset Contents
* `loan_size`: The loaned amount provided to the borrower.
* `interest_rate`: The interest rate of the loan.
* `borrower_income`: Annual income of the borrower.
* `debt_to_income`: Debt to Income ration of the borrower.
* `num_of_accounts`: The number of accounts the borrower has.
* `derogatory_marks`: Amount of derogatory marks on the borrower credit report.
* `total_debt`: The amount of debt a borrower has.
* `loan_status`: The status of the loan, where 0 indicates a loan is healthy and 1 indicates a high risk of defaulting.

## Machine Learning Stages
Describe the stages of the machine learning process you went through as part of this analysis.
Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).

First, I imported the data and placed it into DataFrame (`lending_df`). After this step, I need to split the data into Labels (y) and Features (X).

```python
y = lending_df['loan_status']

X = lending_df.drop(columns='loan_status')
```

Split the data randomly into training and testing data assigning a `random_state` of 1.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
X_train.shape

Result
(58152, 7)
```
Next we need to fit a logistic regression model using the training data created above. Logistic regression is a method for predicting binary outcomes (1 or 0) from a data set. Using the rest of the data on the borrower, the model will be able to predict the credit worthiness of the customer by predicting loan_status and checking against the actual loan_status.

```python
# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
classifier = LogisticRegression(solver='lbfgs', max_iter=200, random_state=1)

# Fit the model using training data
classifier.fit(X_train, y_train)

# Make predictions using test data
predictions = classifier.predict(X_test)
results = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)
results.head(10)
Index Prediction Actual
0	    0	    0
1	    0	    1
2	    0	    0
3	    0	    0
4	    0	    0
5	    0	    0
6	    0	    0
7	    0	    0
8	    0	    0
9	    0	    0
```


Create a confusion matrix. This reveals the number of true negatives/positives for each class and compares them with the predicted values. For example, you see the predicted 0 and actual 0 is 18,679 and Predicted 1 and Actual 1 is 558.

Predicted is along the x-axis of the matrix, Actual is along the y-axis.

Judging by the results we can see that the model is very accurate.

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

Result
array([[18679,    80],
       [   67,   558]], dtype=int64)
```

Create a classification report which will allow us to evaluate the number of predicted occurrences. This is split into precision, recall, and accuracy.

* `precision`: The ratio of correctly predicted positive observations to the total predicted positive observations.
* `recall`: The ratio of correctly predicted positives obersations to all predicted observations.
* `accuracy`:  is how often the model is correct. This is calculated as the ratio of correctly predicted observations to the total number of observations.

```python

from sklearn.metrics import classification_report
target_names = ["healthy loan", "high-risk loan"]
print(classification_report(y_test, predictions, target_names=target_names))

                precision    recall  f1-score   support

  healthy loan       1.00      1.00      1.00     18759
high-risk loan       0.87      0.89      0.88       625

      accuracy                           0.99     19384
     macro avg       0.94      0.94      0.94     19384
  weighted avg       0.99      0.99      0.99     19384
```

## Results

* `precision`: The model accurately predicted healthy loans with 100% precision, and high-risk loans with 87% precision
* `recall`: The model accurately predicted healthy loans with 100% recall, and high-risk loans with 89% precision
* `accuracy`:  The model is 94% accurate with is predictions.

## Summary

* This model is excellent for predicting borrowers who will maintain a healthy loan. However, it is us up the lender to assess the risk of the inaccuracy of the model. In this case, the accuracy is very high, but there is a margin for error in determining borrowers who may default.
