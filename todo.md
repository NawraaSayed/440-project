# Random Forest To-Do List  
## Credit Risk Analysis / Loan Default Prediction

**Project:** ITCS440 Intelligent Systems – Project 8: ML Applications  
**Idea:** Credit Analysis  
**Dataset:** Loan Dataset  
**Main Model:** Random Forest Classifier  
**Target Column:** `Current_loan_status`

---

## 1. Project Goal

Build a machine learning model that predicts whether a loan customer will:

- `DEFAULT`
- `NO DEFAULT`

This is a **binary classification problem**.

The model will use customer and loan information such as:

- Customer age
- Customer income
- Home ownership
- Employment duration
- Loan intent
- Loan grade
- Loan amount
- Interest rate
- Loan term
- Historical default
- Credit history length

---

## 2. Required Libraries

Install the needed libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Import them in Python:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
```

---

## 3. Load and Understand the Dataset

### To-Do

```python
df = pd.read_csv("LoanDataset - LoansDatasest.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df["Current_loan_status"].value_counts())
```

### What to check

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Number of `DEFAULT` and `NO DEFAULT` cases

### Report Explanation

The dataset contains customer and loan information. The target variable is `Current_loan_status`, which shows whether the customer defaulted or did not default. The dataset includes both numerical and categorical features, so preprocessing is required before applying the Random Forest model.

---

## 4. Data Cleaning

### 4.1 Remove Missing Target Rows

```python
df = df.dropna(subset=["Current_loan_status"])
```

### 4.2 Drop Customer ID

```python
df = df.drop(columns=["customer_id"])
```

`customer_id` should not be used because it is only an identifier and does not help prediction.

### 4.3 Clean Money Columns

Convert money columns from text to numeric values.

```python
df["customer_income"] = (
    df["customer_income"]
    .astype(str)
    .str.replace("£", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df["loan_amnt"] = (
    df["loan_amnt"]
    .astype(str)
    .str.replace("£", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)
```

### 4.4 Convert Target Column to Numbers

```python
df["Current_loan_status"] = df["Current_loan_status"].map({
    "NO DEFAULT": 0,
    "DEFAULT": 1
})
```

Meaning:

- `0 = NO DEFAULT`
- `1 = DEFAULT`

---

## 5. Exploratory Data Analysis Graphs

These graphs should be included in the report or presentation.

---

### Graph 1: Target Distribution

Purpose: Show how many customers defaulted and did not default.

```python
sns.countplot(data=df, x="Current_loan_status")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
plt.ylabel("Count")
plt.show()
```

### Explanation

This graph shows that the dataset has more `NO DEFAULT` cases than `DEFAULT` cases. Therefore, F1 Score is important because accuracy alone may be misleading.

---

### Graph 2: Loan Grade vs Default

Purpose: Show if risky loan grades have more defaults.

```python
sns.countplot(data=df, x="loan_grade", hue="Current_loan_status")
plt.title("Loan Grade vs Loan Default")
plt.xlabel("Loan Grade")
plt.ylabel("Count")
plt.show()
```

### Explanation

This graph shows the relationship between loan grade and default status. Higher-risk loan grades are expected to have more default cases.

---

### Graph 3: Interest Rate by Loan Status

Purpose: Show whether default loans have higher interest rates.

```python
sns.boxplot(data=df, x="Current_loan_status", y="loan_int_rate")
plt.title("Interest Rate by Loan Status")
plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
plt.ylabel("Interest Rate")
plt.show()
```

### Explanation

This boxplot compares loan interest rates between default and no-default customers. Higher interest rates may indicate higher loan risk.

---

### Graph 4: Customer Income by Loan Status

Purpose: Show if income has a relationship with default.

```python
sns.boxplot(data=df, x="Current_loan_status", y="customer_income")
plt.ylim(0, df["customer_income"].quantile(0.95))
plt.title("Customer Income by Loan Status")
plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
plt.ylabel("Customer Income")
plt.show()
```

### Explanation

This graph compares customer income between default and no-default customers. The limit is used to reduce the effect of very large outliers.

---

### Graph 5: Loan Amount by Loan Status

Purpose: Show whether larger loans have more risk.

```python
sns.boxplot(data=df, x="Current_loan_status", y="loan_amnt")
plt.title("Loan Amount by Loan Status")
plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
plt.ylabel("Loan Amount")
plt.show()
```

### Explanation

This graph compares loan amount between default and no-default customers.

---

### Graph 6: Correlation Heatmap

Purpose: Show relationships between numerical features.

```python
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

### Explanation

The correlation heatmap shows the relationship between numerical features. It helps identify which numerical variables may be related to loan default.

---

## 6. Prepare Features and Target

### 6.1 Separate X and y

```python
X = df.drop(columns=["Current_loan_status"])
y = df["Current_loan_status"]
```

### 6.2 Define Numerical Features

```python
numeric_features = [
    "customer_age",
    "customer_income",
    "employment_duration",
    "loan_amnt",
    "loan_int_rate",
    "term_years",
    "cred_hist_length"
]
```

### 6.3 Define Categorical Features

```python
categorical_features = [
    "home_ownership",
    "loan_intent",
    "loan_grade",
    "historical_default"
]
```

---

## 7. Preprocessing Pipeline

### Numerical Preprocessing

For numerical columns:

- Fill missing values using median

### Categorical Preprocessing

For categorical columns:

- Fill missing values using the most frequent value
- Convert text categories using One-Hot Encoding

### Code

```python
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
```

---

## 8. Train-Test Split

Use stratify because the dataset has more `NO DEFAULT` cases than `DEFAULT` cases.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Explanation

Stratify keeps the same percentage of `DEFAULT` and `NO DEFAULT` cases in both training and testing data.

---

## 9. Build the Random Forest Model

```python
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    ))
])
```

### Important Parameters

| Parameter | Meaning |
|---|---|
| `n_estimators=200` | The model uses 200 decision trees |
| `max_depth=None` | Trees can grow until fully expanded |
| `random_state=42` | Makes results reproducible |
| `class_weight="balanced"` | Helps with class imbalance |

---

## 10. Train the Model

```python
rf_model.fit(X_train, y_train)
```

---

## 11. Make Predictions

```python
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
```

`y_pred` is used for classification results.  
`y_prob` is used for the ROC curve.

---

## 12. Evaluate the Model

### 12.1 Accuracy, F1 Score, and MSE

```python
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Random Forest Results")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("MSE:", mse)
```

### 12.2 Classification Report

```python
print(classification_report(y_test, y_pred, target_names=["NO DEFAULT", "DEFAULT"]))
```

### Metrics to Discuss

| Metric | Meaning |
|---|---|
| Accuracy | Percentage of correct predictions |
| F1 Score | Balance between precision and recall |
| MSE | Average squared prediction error |
| Precision | How many predicted defaults were actually defaults |
| Recall | How many actual defaults were detected |

### Most Important Metric

For this project, the most important metric is:

```text
F1 Score for DEFAULT
```

because in credit analysis, detecting customers who may default is very important.

---

## 13. Evaluation Graphs

---

### Graph 7: Confusion Matrix

Purpose: Show correct and wrong predictions.

```python
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NO DEFAULT", "DEFAULT"]
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()
```

### Explanation

The confusion matrix shows how many customers were correctly classified as `DEFAULT` or `NO DEFAULT`. It also shows incorrect predictions, such as customers predicted as safe but actually defaulted.

---

### Graph 8: ROC Curve

Purpose: Show how well the model separates `DEFAULT` and `NO DEFAULT`.

```python
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Random Forest ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

### Explanation

The ROC curve shows the model’s ability to distinguish between `DEFAULT` and `NO DEFAULT` classes. A higher AUC means better classification performance.

---

### Graph 9: Feature Importance

Purpose: Show which features affected the prediction the most.

```python
rf_classifier = rf_model.named_steps["classifier"]
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

importances = rf_classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df.head(15))
```

Graph:

```python
top_features = feature_importance_df.head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x="Importance", y="Feature")
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
```

### Explanation

This graph shows the most important features used by the Random Forest model. Features such as loan interest rate, loan grade, income, loan amount, and historical default are expected to strongly affect loan default prediction.

---

## 14. Optional Model Tuning

Use GridSearchCV to improve the Random Forest model.

```python
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)
```

Use the best model:

```python
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

print("Tuned Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("F1 Score:", f1_score(y_test, y_pred_best))
print("MSE:", mean_squared_error(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
```

---

## 15. Results Table

Create a table for the Random Forest result.

```python
results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "F1 Score": [f1],
    "MSE": [mse]
})

print(results)
```

Later, compare Random Forest with:

- Logistic Regression
- Decision Tree
- SVM
- Neural Network

Final comparison table example:

| Model | Accuracy | F1 Score | MSE |
|---|---:|---:|---:|
| Logistic Regression | ... | ... | ... |
| Decision Tree | ... | ... | ... |
| SVM | ... | ... | ... |
| Neural Network | ... | ... | ... |
| Random Forest | ... | ... | ... |

---

## 16. Report Structure

### 1. Problem Definition

The aim of this project is to predict whether a loan customer will default or not using machine learning. The problem is a binary classification problem where the target variable is `Current_loan_status`.

### 2. Dataset Description

Mention:

- Number of rows
- Number of columns
- Target column
- Numerical features
- Categorical features
- Missing values

### 3. Data Preprocessing

Mention that you:

- Removed `customer_id`
- Removed rows with missing `Current_loan_status`
- Converted income and loan amount to numeric
- Filled missing numerical values using median
- Filled missing categorical values using most frequent value
- Encoded categorical variables using OneHotEncoder
- Split data into training and testing sets

### 4. Random Forest Model

Random Forest is an ensemble learning model that builds many decision trees and combines their predictions. It is suitable for this project because it can handle complex relationships and performs well on classification problems.

### 5. Evaluation Metrics

Explain:

- Accuracy
- F1 Score
- MSE
- Confusion Matrix
- ROC Curve

### 6. Results

Include:

- Accuracy
- F1 Score
- MSE
- Confusion matrix
- ROC curve
- Feature importance graph

### 7. Conclusion

The Random Forest model performed well for credit risk analysis because it achieved strong classification results and was able to identify important features affecting loan default. Since the dataset is imbalanced, F1 Score was considered more important than accuracy alone.

---

## 17. Final Checklist

**Progress:** 32 / 32 tasks completed = **100.00% done**

### Completed Items Summary

- Dataset loaded from `LoanDataset - LoansDatasest.csv`
- Shape: 32,586 rows and 13 columns
- Columns: `customer_id`, `customer_age`, `customer_income`, `home_ownership`, `employment_duration`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `term_years`, `historical_default`, `cred_hist_length`, `Current_loan_status`
- Missing values found in `customer_id` (3), `employment_duration` (895), `loan_amnt` (1), `loan_int_rate` (3,116), `historical_default` (20,737), and `Current_loan_status` (4)
- Target distribution: `NO DEFAULT` = 25,742, `DEFAULT` = 6,840, missing = 4
- Cleaned dataset saved as `loan_dataset_cleaned.csv`
- Removed `customer_id`, removed 4 rows with missing `Current_loan_status`, cleaned `customer_income` and `loan_amnt`, and converted target values to `0 = NO DEFAULT` and `1 = DEFAULT`
- Target distribution graph saved as `figures/target_distribution.svg`
- Loan grade vs default graph saved as `figures/loan_grade_vs_default.svg`
- Interest rate by loan status boxplot saved as `figures/interest_rate_by_loan_status.svg`
- Customer income by loan status boxplot saved as `figures/customer_income_by_loan_status.svg` with the y-axis capped at the 95th percentile to make the central distribution readable
- Loan amount by loan status boxplot saved as `figures/loan_amount_by_loan_status.svg` with the y-axis capped at the 99th percentile because extreme outliers made the original graph unclear
- Correlation heatmap saved as `figures/correlation_heatmap.svg`
- Features and target separated using `Current_loan_status` as the target
- Numerical features defined: `customer_age`, `customer_income`, `employment_duration`, `loan_amnt`, `loan_int_rate`, `term_years`, `cred_hist_length`
- Categorical features defined: `home_ownership`, `loan_intent`, `loan_grade`
- `historical_default` excluded from the final model because its missing values perfectly identify `NO DEFAULT` rows, which creates target leakage and inflates accuracy
- Preprocessing pipeline built with median imputation for numerical features and most-frequent imputation plus one-hot encoding for categorical features
- Stratified train/test split completed with 26,065 training rows and 6,517 testing rows
- Preprocessing summary saved as `preprocessing_summary.json`
- Random Forest pipeline built using preprocessing plus a regularized `RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, class_weight="balanced")`
- Random Forest trained on 26,065 training rows
- Predictions and probabilities generated for 6,517 testing rows
- Final Random Forest metrics saved in `preprocessing_summary.json` after removing leakage: Accuracy = 0.9190, F1 Score = 0.7901, MSE = 0.0810, ROC AUC = 0.9313
- Classification report saved as `classification_report.txt`
- Confusion matrix graph saved as `figures/confusion_matrix.svg`
- ROC curve graph saved as `figures/roc_curve.svg`
- Feature importance graph saved as `figures/feature_importance.svg`; full feature importances saved as `feature_importance.csv`
- Optional focused GridSearchCV tuning completed and saved as `tuning_summary.json`; tuned test F1 Score = 0.7888 and tuned ROC AUC = 0.9322
- Final results table saved as `results_table.csv`
- Report explanation written in `report_explanation.md`

```text
[x] Load dataset
[x] Check shape, columns, data types
[x] Check missing values
[x] Check target distribution
[x] Drop customer_id
[x] Drop rows with missing Current_loan_status
[x] Clean customer_income
[x] Clean loan_amnt
[x] Convert target: DEFAULT = 1, NO DEFAULT = 0
[x] Create target distribution graph
[x] Create loan grade vs default graph
[x] Create interest rate boxplot
[x] Create income boxplot
[x] Create loan amount boxplot
[x] Create correlation heatmap
[x] Split X and y
[x] Define numerical columns
[x] Define categorical columns
[x] Build preprocessing pipeline
[x] Split train/test using stratify
[x] Build Random Forest pipeline
[x] Train Random Forest
[x] Predict test data
[x] Calculate Accuracy
[x] Calculate F1 Score
[x] Calculate MSE
[x] Print classification report
[x] Create confusion matrix graph
[x] Create ROC curve
[x] Create feature importance graph
[x] Optional: tune model using GridSearchCV
[x] Save final results table
[x] Write report explanation
```

---

## 18. Most Important Graphs to Include

Use these in the report and presentation:

1. Target distribution
2. Loan grade vs default
3. Interest rate by loan status
4. Customer income by loan status
5. Confusion matrix
6. ROC curve
7. Feature importance

These graphs are enough for a strong report and presentation.
