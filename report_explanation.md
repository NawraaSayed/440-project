# Random Forest Credit Risk Report

## Problem Definition

The goal of this project is to predict whether a loan customer will default or not. This is a binary classification problem using the target column `Current_loan_status`, where `0` means `NO DEFAULT` and `1` means `DEFAULT`.

## Dataset

The cleaned dataset contains 32,582 rows and 12 columns. The final model uses 8 input features: 6 numerical features and 2 categorical features.

The target classes are imbalanced:

- `NO DEFAULT`: 25,742 records
- `DEFAULT`: 6,840 records

Because there are more `NO DEFAULT` records than `DEFAULT` records, accuracy alone is not enough to judge the model. F1 score, recall, the confusion matrix, and ROC AUC are also important.

## Preprocessing

Rows with missing target values were removed, `customer_id` was dropped, money columns were converted to numeric values, and the target labels were converted to `0` and `1`. Numerical missing values are filled with the median. Categorical missing values are filled with the most frequent category and then one-hot encoded. The `historical_default` column was excluded because its missing values perfectly identify `NO DEFAULT` records in this dataset. The `loan_grade` and `loan_int_rate` columns were also excluded from the final model because they are likely assigned during underwriting/pricing and can make the result look stronger than a pre-approval risk model should be.

## Model

The main model is a regularized Random Forest classifier with 200 trees, `max_depth=12`, `min_samples_leaf=10`, `min_samples_split=20`, `random_state=42`, and `class_weight="balanced"` to reduce the effect of class imbalance and overfitting.

## Results

- Accuracy: 0.8215 (82.15%)
- F1 Score for DEFAULT: 0.6122
- MSE: 0.1785
- ROC AUC: 0.8488

The tuned Random Forest produced a very similar result, with F1 Score = 0.6122 and ROC AUC = 0.8488. Since the improvement was very small, the regularized Random Forest result is a suitable final model.

## Overfitting Check

The final model was checked by comparing train and test performance and by running 3-fold cross-validation on the training data.

- Assessment: No major overfitting signal
- Train Accuracy: 0.8484
- Test Accuracy: 0.8215
- Accuracy Gap: 0.0269
- Train F1 Score for `DEFAULT`: 0.6726
- Test F1 Score for `DEFAULT`: 0.6122
- F1 Gap: 0.0604
- Cross-validation F1 mean: 0.6007
- Cross-validation F1 standard deviation: 0.0069

The train score is higher than the test score, which is normal for a Random Forest, but the gap is not large enough to indicate a serious overfitting problem. The cross-validation F1 score is also close to the held-out test F1 score, so the model appears to generalize reasonably after removing the leakage feature.

## Red-Flag Checks

The original high-accuracy result was checked against several common failure modes:

- Majority-class baseline accuracy: 0.7901
- Exact duplicate rows in full dataset: 142
- Exact duplicate rows crossing train/test split: 68
- Accuracy with `loan_grade` and `loan_int_rate`: 0.8780
- Accuracy without `loan_grade` but with `loan_int_rate`: 0.8737
- Final accuracy without `loan_grade` or `loan_int_rate`: 0.8219
- Accuracy after removing exact duplicate rows: 0.8146
- Accuracy with the known leakage column `historical_default`: 0.9538
- Shuffled-target ROC AUC: 0.5040

These checks show that the earlier 90% result was not caused by exact duplicate rows or a broken validation setup. The real problem was feature timing: `loan_grade` and `loan_int_rate` carry lender risk/pricing information. The final model excludes them, so the reported result is lower but more defensible for a pre-approval credit-risk model.

Classification report:

```text
              precision    recall  f1-score   support

  NO DEFAULT       0.91      0.86      0.88      5149
     DEFAULT       0.56      0.67      0.61      1368

    accuracy                           0.82      6517
   macro avg       0.74      0.77      0.75      6517
weighted avg       0.84      0.82      0.83      6517

```

## Confusion Matrix Interpretation

- Correctly predicted `NO DEFAULT`: 4436
- Incorrectly predicted `DEFAULT` for actual `NO DEFAULT`: 713
- Incorrectly predicted `NO DEFAULT` for actual `DEFAULT`: 450
- Correctly predicted `DEFAULT`: 918

The most important error in credit risk is predicting `NO DEFAULT` when the customer actually defaults. The model made 450 of these errors on the test set. This is why the `DEFAULT` recall of 0.67 should be discussed in the report.

## Most Important Features

- `num__customer_income`: 0.3360
- `num__loan_amnt`: 0.1784
- `cat__home_ownership_RENT`: 0.1133
- `cat__home_ownership_MORTGAGE`: 0.0632
- `num__customer_age`: 0.0468
- `num__employment_duration`: 0.0424
- `num__term_years`: 0.0396
- `num__cred_hist_length`: 0.0356
- `cat__home_ownership_OWN`: 0.0350
- `cat__loan_intent_DEBTCONSOLIDATION`: 0.0293

## Conclusion

After removing leakage and proxy-leakage features, the Random Forest model gives a more realistic result with no major overfitting signal. The final accuracy is 82.15%, and the F1 score for the `DEFAULT` class is 0.6122. The stricter model is less impressive than the earlier 90% result, but it is more appropriate if predictions are made before loan grade and interest rate are assigned. The feature importance results show that customer income, loan amount, age, home ownership, loan intent, term length, and credit history length are important predictors for credit risk.
