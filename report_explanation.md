# Random Forest Credit Risk Report

## Problem Definition

The goal of this project is to predict whether a loan customer will default or not. This is a binary classification problem using the target column `Current_loan_status`, where `0` means `NO DEFAULT` and `1` means `DEFAULT`.

## Dataset

The cleaned dataset contains 32,582 rows and 12 columns. The final model uses 10 input features: 7 numerical features and 3 categorical features.

The target classes are imbalanced:

- `NO DEFAULT`: 25,742 records
- `DEFAULT`: 6,840 records

Because there are more `NO DEFAULT` records than `DEFAULT` records, accuracy alone is not enough to judge the model. F1 score, recall, the confusion matrix, and ROC AUC are also important.

## Preprocessing

Rows with missing target values were removed, `customer_id` was dropped, money columns were converted to numeric values, and the target labels were converted to `0` and `1`. Numerical missing values are filled with the median. Categorical missing values are filled with the most frequent category and then one-hot encoded. The `historical_default` column was excluded because its missing values perfectly identify `NO DEFAULT` records in this dataset, which would cause target leakage.

## Model

The main model is a regularized Random Forest classifier with 200 trees, `max_depth=16`, `min_samples_leaf=5`, `min_samples_split=10`, `random_state=42`, and `class_weight="balanced"` to reduce the effect of class imbalance and overfitting.

## Results

- Accuracy: 0.9044 (90.44%)
- F1 Score for DEFAULT: 0.7704
- MSE: 0.0956
- ROC AUC: 0.9274

The tuned Random Forest produced a very similar result, with F1 Score = 0.7704 and ROC AUC = 0.9274. Since the improvement was very small, the regularized Random Forest result is a suitable final model.

## Overfitting Check

The final model was checked by comparing train and test performance and by running 3-fold cross-validation on the training data.

- Assessment: No major overfitting signal
- Train Accuracy: 0.9409
- Test Accuracy: 0.9044
- Accuracy Gap: 0.0365
- Train F1 Score for `DEFAULT`: 0.8605
- Test F1 Score for `DEFAULT`: 0.7704
- F1 Gap: 0.0901
- Cross-validation F1 mean: 0.7601
- Cross-validation F1 standard deviation: 0.0021

The train score is higher than the test score, which is normal for a Random Forest, but the gap is not large enough to indicate a serious overfitting problem. The cross-validation F1 score is also close to the held-out test F1 score, so the model appears to generalize reasonably after removing the leakage feature.

Classification report:

```text
              precision    recall  f1-score   support

  NO DEFAULT       0.94      0.94      0.94      5149
     DEFAULT       0.78      0.76      0.77      1368

    accuracy                           0.90      6517
   macro avg       0.86      0.85      0.86      6517
weighted avg       0.90      0.90      0.90      6517

```

## Confusion Matrix Interpretation

- Correctly predicted `NO DEFAULT`: 4849
- Incorrectly predicted `DEFAULT` for actual `NO DEFAULT`: 300
- Incorrectly predicted `NO DEFAULT` for actual `DEFAULT`: 323
- Correctly predicted `DEFAULT`: 1045

The most important error in credit risk is predicting `NO DEFAULT` when the customer actually defaults. The model made 323 of these errors on the test set. This is why the `DEFAULT` recall of 0.76 should be discussed in the report.

## Most Important Features

- `num__customer_income`: 0.2205
- `num__loan_int_rate`: 0.1491
- `num__loan_amnt`: 0.1086
- `cat__home_ownership_RENT`: 0.0657
- `cat__loan_grade_A`: 0.0641
- `num__customer_age`: 0.0506
- `cat__loan_grade_D`: 0.0444
- `num__cred_hist_length`: 0.0421
- `num__employment_duration`: 0.0393
- `num__term_years`: 0.0340

## Conclusion

After removing the leakage feature and using a more regularized Random Forest, the model gives a more realistic result with no major overfitting signal. The final accuracy is 90.44%, and the F1 score for the `DEFAULT` class is 0.7704. The model is better at identifying `NO DEFAULT` customers than `DEFAULT` customers, but it still detects most default cases. The feature importance results show that customer income, interest rate, loan amount, age, home ownership, and loan grade are important predictors for credit risk.
