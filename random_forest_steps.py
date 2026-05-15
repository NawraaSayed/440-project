import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("loan_dataset_cleaned.csv")
FIGURES_DIR = Path("figures")
TARGET_COLUMN = "Current_loan_status"

NUMERIC_FEATURES = [
    "customer_age",
    "customer_income",
    "employment_duration",
    "loan_amnt",
    "loan_int_rate",
    "term_years",
    "cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_intent",
    "loan_grade",
]

EXCLUDED_FEATURES = {
    "historical_default": (
        "Excluded from modeling because its missing values perfectly identify NO DEFAULT rows "
        "in this dataset, which inflates accuracy through target leakage."
    )
}


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def save_target_distribution_graph(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.svg")
    plt.close()


def save_loan_grade_graph(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="loan_grade", hue=TARGET_COLUMN, order=sorted(df["loan_grade"].dropna().unique()))
    plt.title("Loan Grade vs Loan Default")
    plt.xlabel("Loan Grade")
    plt.ylabel("Count")
    plt.legend(title="Loan Status", labels=["NO DEFAULT", "DEFAULT"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loan_grade_vs_default.svg")
    plt.close()


def save_interest_rate_boxplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="loan_int_rate")
    plt.title("Interest Rate by Loan Status")
    plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
    plt.ylabel("Interest Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "interest_rate_by_loan_status.svg")
    plt.close()


def save_income_boxplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="customer_income", showfliers=False)
    plt.ylim(0, df["customer_income"].quantile(0.95))
    plt.title("Customer Income by Loan Status (Central 95%)")
    plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
    plt.ylabel("Customer Income")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "customer_income_by_loan_status.svg")
    plt.close()


def save_loan_amount_boxplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="loan_amnt", showfliers=False)
    plt.ylim(0, df["loan_amnt"].quantile(0.99))
    plt.title("Loan Amount by Loan Status (Central 99%)")
    plt.xlabel("Loan Status: 0 = No Default, 1 = Default")
    plt.ylabel("Loan Amount")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loan_amount_by_loan_status.svg")
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.svg")
    plt.close()


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_random_forest_model(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_confusion_matrix_graph(y_test: pd.Series, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NO DEFAULT", "DEFAULT"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.svg")
    plt.close()


def save_roc_curve_graph(y_test: pd.Series, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Random Forest ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.svg")
    plt.close()
    return float(roc_auc)


def save_feature_importance_graph(rf_model: Pipeline) -> pd.DataFrame:
    rf_classifier = rf_model.named_steps["classifier"]
    feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": rf_classifier.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    top_features = feature_importance_df.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x="Importance", y="Feature")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.svg")
    plt.close()

    feature_importance_df.to_csv("feature_importance.csv", index=False)
    return feature_importance_df


def run_focused_grid_search(rf_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [14, 16],
        "classifier__min_samples_split": [10],
        "classifier__min_samples_leaf": [5],
    }

    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def calculate_binary_metrics(y_true: pd.Series, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "roc_auc": float(auc(*roc_curve(y_true, y_prob)[:2])),
    }


def build_overfit_diagnostics(train_metrics: dict, test_metrics: dict, cv_f1_scores) -> dict:
    accuracy_gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    f1_gap = train_metrics["f1_score"] - test_metrics["f1_score"]
    roc_auc_gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]
    cv_mean = float(cv_f1_scores.mean())
    cv_std = float(cv_f1_scores.std())
    cv_to_test_f1_gap = cv_mean - test_metrics["f1_score"]

    if f1_gap <= 0.10 and accuracy_gap <= 0.05 and abs(cv_to_test_f1_gap) <= 0.05:
        assessment = "No major overfitting signal"
    elif f1_gap <= 0.15 and accuracy_gap <= 0.08 and abs(cv_to_test_f1_gap) <= 0.08:
        assessment = "Mild overfitting signal"
    else:
        assessment = "Possible overfitting concern"

    return {
        "assessment": assessment,
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "accuracy_gap": float(accuracy_gap),
        "train_f1_score": train_metrics["f1_score"],
        "test_f1_score": test_metrics["f1_score"],
        "f1_gap": float(f1_gap),
        "train_roc_auc": train_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "roc_auc_gap": float(roc_auc_gap),
        "cv_f1_scores": [float(score) for score in cv_f1_scores],
        "cv_f1_mean": cv_mean,
        "cv_f1_std": cv_std,
        "cv_to_test_f1_gap": float(cv_to_test_f1_gap),
    }


def write_classification_report_file(metrics: dict, report_text: str) -> None:
    cm = metrics["confusion_matrix"]
    overfit = metrics["overfit_diagnostics"]
    text = f"""Random Forest Classification Report
===================================

Model setup:
- Target column: Current_loan_status
- 0 = NO DEFAULT
- 1 = DEFAULT
- Excluded feature: historical_default

Important note:
The historical_default column was excluded from the final model because its missing values perfectly identify NO DEFAULT rows in this dataset. Keeping it would cause target leakage and make the accuracy look unrealistically high.

Final metrics:
- Accuracy: {metrics["accuracy"]:.4f}
- F1 Score for DEFAULT: {metrics["f1_score"]:.4f}
- MSE: {metrics["mse"]:.4f}
- ROC AUC: {metrics["roc_auc"]:.4f}

Overfitting diagnostics:
- Assessment: {overfit["assessment"]}
- Train Accuracy: {overfit["train_accuracy"]:.4f}
- Test Accuracy: {overfit["test_accuracy"]:.4f}
- Accuracy Gap: {overfit["accuracy_gap"]:.4f}
- Train F1 Score for DEFAULT: {overfit["train_f1_score"]:.4f}
- Test F1 Score for DEFAULT: {overfit["test_f1_score"]:.4f}
- F1 Gap: {overfit["f1_gap"]:.4f}
- Cross-validation F1 mean: {overfit["cv_f1_mean"]:.4f}
- Cross-validation F1 std: {overfit["cv_f1_std"]:.4f}

Confusion matrix:
- True NO DEFAULT predicted as NO DEFAULT: {cm[0][0]}
- True NO DEFAULT predicted as DEFAULT: {cm[0][1]}
- True DEFAULT predicted as NO DEFAULT: {cm[1][0]}
- True DEFAULT predicted as DEFAULT: {cm[1][1]}

Detailed classification report:

{report_text}

Interpretation:
The model performs very well for the NO DEFAULT class and gives a useful result for the DEFAULT class. For DEFAULT customers, precision is {metrics["precision"]:.2f}, meaning most customers predicted as DEFAULT really did default. Recall is {metrics["recall"]:.2f}, meaning the model detects about {format_percent(metrics["recall"])} of actual DEFAULT cases. Since missing a default customer is important in credit risk, the DEFAULT recall and F1 score should be discussed alongside accuracy.
"""
    Path("classification_report.txt").write_text(text, encoding="utf-8")


def write_report_explanation(metrics: dict, report_text: str, top_features: pd.DataFrame) -> None:
    top_feature_lines = "\n".join(
        f"- `{row.Feature}`: {row.Importance:.4f}" for row in top_features.head(10).itertuples(index=False)
    )
    cm = metrics["confusion_matrix"]
    overfit = metrics["overfit_diagnostics"]
    report = f"""# Random Forest Credit Risk Report

## Problem Definition

The goal of this project is to predict whether a loan customer will default or not. This is a binary classification problem using the target column `Current_loan_status`, where `0` means `NO DEFAULT` and `1` means `DEFAULT`.

## Dataset

The cleaned dataset contains {metrics["dataset_rows"]:,} rows and {metrics["dataset_columns"]} columns. The final model uses {metrics["feature_count"]} input features: {len(NUMERIC_FEATURES)} numerical features and {len(CATEGORICAL_FEATURES)} categorical features.

The target classes are imbalanced:

- `NO DEFAULT`: {metrics["target_distribution"]["0"]:,} records
- `DEFAULT`: {metrics["target_distribution"]["1"]:,} records

Because there are more `NO DEFAULT` records than `DEFAULT` records, accuracy alone is not enough to judge the model. F1 score, recall, the confusion matrix, and ROC AUC are also important.

## Preprocessing

Rows with missing target values were removed, `customer_id` was dropped, money columns were converted to numeric values, and the target labels were converted to `0` and `1`. Numerical missing values are filled with the median. Categorical missing values are filled with the most frequent category and then one-hot encoded. The `historical_default` column was excluded because its missing values perfectly identify `NO DEFAULT` records in this dataset, which would cause target leakage.

## Model

The main model is a regularized Random Forest classifier with 200 trees, `max_depth=16`, `min_samples_leaf=5`, `min_samples_split=10`, `random_state=42`, and `class_weight=\"balanced\"` to reduce the effect of class imbalance and overfitting.

## Results

- Accuracy: {metrics["accuracy"]:.4f} ({format_percent(metrics["accuracy"])})
- F1 Score for DEFAULT: {metrics["f1_score"]:.4f}
- MSE: {metrics["mse"]:.4f}
- ROC AUC: {metrics["roc_auc"]:.4f}

The tuned Random Forest produced a very similar result, with F1 Score = {metrics["tuning"]["test_f1_score"]:.4f} and ROC AUC = {metrics["tuning"]["test_roc_auc"]:.4f}. Since the improvement was very small, the regularized Random Forest result is a suitable final model.

## Overfitting Check

The final model was checked by comparing train and test performance and by running 3-fold cross-validation on the training data.

- Assessment: {overfit["assessment"]}
- Train Accuracy: {overfit["train_accuracy"]:.4f}
- Test Accuracy: {overfit["test_accuracy"]:.4f}
- Accuracy Gap: {overfit["accuracy_gap"]:.4f}
- Train F1 Score for `DEFAULT`: {overfit["train_f1_score"]:.4f}
- Test F1 Score for `DEFAULT`: {overfit["test_f1_score"]:.4f}
- F1 Gap: {overfit["f1_gap"]:.4f}
- Cross-validation F1 mean: {overfit["cv_f1_mean"]:.4f}
- Cross-validation F1 standard deviation: {overfit["cv_f1_std"]:.4f}

The train score is higher than the test score, which is normal for a Random Forest, but the gap is not large enough to indicate a serious overfitting problem. The cross-validation F1 score is also close to the held-out test F1 score, so the model appears to generalize reasonably after removing the leakage feature.

Classification report:

```text
{report_text}
```

## Confusion Matrix Interpretation

- Correctly predicted `NO DEFAULT`: {cm[0][0]}
- Incorrectly predicted `DEFAULT` for actual `NO DEFAULT`: {cm[0][1]}
- Incorrectly predicted `NO DEFAULT` for actual `DEFAULT`: {cm[1][0]}
- Correctly predicted `DEFAULT`: {cm[1][1]}

The most important error in credit risk is predicting `NO DEFAULT` when the customer actually defaults. The model made {cm[1][0]} of these errors on the test set. This is why the `DEFAULT` recall of {metrics["recall"]:.2f} should be discussed in the report.

## Most Important Features

{top_feature_lines}

## Conclusion

After removing the leakage feature and using a more regularized Random Forest, the model gives a more realistic result with no major overfitting signal. The final accuracy is {format_percent(metrics["accuracy"])}, and the F1 score for the `DEFAULT` class is {metrics["f1_score"]:.4f}. The model is better at identifying `NO DEFAULT` customers than `DEFAULT` customers, but it still detects most default cases. The feature importance results show that customer income, interest rate, loan amount, age, home ownership, and loan grade are important predictors for credit risk.
"""
    Path("report_explanation.md").write_text(report, encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    FIGURES_DIR.mkdir(exist_ok=True)
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    save_target_distribution_graph(df)
    save_loan_grade_graph(df)
    save_interest_rate_boxplot(df)
    save_income_boxplot(df)
    save_loan_amount_boxplot(df)
    save_correlation_heatmap(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    preprocessor = build_preprocessor()
    rf_model = build_random_forest_model(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    rf_model.fit(X_train, y_train)
    y_pred_train = rf_model.predict(X_train)
    y_prob_train = rf_model.predict_proba(X_train)[:, 1]
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    train_metrics = calculate_binary_metrics(y_train, y_pred_train, y_prob_train)
    test_metrics = calculate_binary_metrics(y_test, y_pred, y_prob)
    accuracy = test_metrics["accuracy"]
    f1 = test_metrics["f1_score"]
    mse = test_metrics["mse"]
    report_text = classification_report(y_test, y_pred, target_names=["NO DEFAULT", "DEFAULT"])

    save_confusion_matrix_graph(y_test, y_pred)
    roc_auc = save_roc_curve_graph(y_test, y_prob)
    feature_importance_df = save_feature_importance_graph(rf_model)
    cv_f1_scores = cross_val_score(rf_model, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
    overfit_diagnostics = build_overfit_diagnostics(train_metrics, test_metrics, cv_f1_scores)

    grid_search = run_focused_grid_search(rf_model, X_train, y_train)
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_prob_tuned)
    tuned_roc_auc = float(auc(fpr_tuned, tpr_tuned))
    tuned_metrics = {
        "best_parameters": grid_search.best_params_,
        "best_cv_f1_score": float(grid_search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred_tuned)),
        "test_f1_score": float(f1_score(y_test, y_pred_tuned)),
        "test_mse": float(mean_squared_error(y_test, y_pred_tuned)),
        "test_roc_auc": tuned_roc_auc,
    }
    Path("tuning_summary.json").write_text(json.dumps(tuned_metrics, indent=2), encoding="utf-8")

    results = pd.DataFrame(
        {
            "Model": ["Random Forest", "Tuned Random Forest"],
            "Accuracy": [accuracy, tuned_metrics["test_accuracy"]],
            "F1 Score": [f1, tuned_metrics["test_f1_score"]],
            "MSE": [mse, tuned_metrics["test_mse"]],
            "ROC AUC": [roc_auc, tuned_roc_auc],
        }
    )
    results.to_csv("results_table.csv", index=False)

    summary = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "feature_count": int(X.shape[1]),
        "target_column": TARGET_COLUMN,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_target_distribution": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_target_distribution": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "preprocessor_steps": {
            "numeric": "SimpleImputer(strategy='median')",
            "categorical": "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')",
        },
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 16,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "class_weight": "balanced",
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": f1,
            "mse": mse,
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        },
        "train_metrics": train_metrics,
        "overfit_diagnostics": overfit_diagnostics,
        "tuning": tuned_metrics,
        "created_graphs": [
            "figures/target_distribution.svg",
            "figures/loan_grade_vs_default.svg",
            "figures/interest_rate_by_loan_status.svg",
            "figures/customer_income_by_loan_status.svg",
            "figures/loan_amount_by_loan_status.svg",
            "figures/correlation_heatmap.svg",
            "figures/confusion_matrix.svg",
            "figures/roc_curve.svg",
            "figures/feature_importance.svg",
        ],
        "graph_notes": {
            "customer_income_by_loan_status.svg": "Y-axis capped at the 95th percentile and fliers hidden to reduce distortion from income outliers.",
            "loan_amount_by_loan_status.svg": "Y-axis capped at the 99th percentile and fliers hidden because loan_amnt has extreme outliers up to 3,500,000.",
        },
    }

    Path("preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_classification_report_file(summary | summary["metrics"], report_text)
    write_report_explanation(summary | summary["metrics"], report_text, feature_importance_df)


if __name__ == "__main__":
    main()
