import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
    "historical_default",
]


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

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    preprocessor = build_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor.fit(X_train)

    summary = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "feature_count": int(X.shape[1]),
        "target_column": TARGET_COLUMN,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_target_distribution": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_target_distribution": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "preprocessor_steps": {
            "numeric": "SimpleImputer(strategy='median')",
            "categorical": "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')",
        },
        "created_graphs": [
            "figures/target_distribution.svg",
            "figures/loan_grade_vs_default.svg",
            "figures/interest_rate_by_loan_status.svg",
            "figures/customer_income_by_loan_status.svg",
            "figures/loan_amount_by_loan_status.svg",
            "figures/correlation_heatmap.svg",
        ],
        "graph_notes": {
            "customer_income_by_loan_status.svg": "Y-axis capped at the 95th percentile and fliers hidden to reduce distortion from income outliers.",
            "loan_amount_by_loan_status.svg": "Y-axis capped at the 99th percentile and fliers hidden because loan_amnt has extreme outliers up to 3,500,000.",
        },
    }

    Path("preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
