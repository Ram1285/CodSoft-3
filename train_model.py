import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("creditcard.csv")
MODEL_PATH = Path("fraud_detection_pipeline.pkl")
METADATA_PATH = Path("feature_metadata.json")
TARGET_COLUMN = "Class"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Place the CodSoft Credit Card Fraud "
            "Detection dataset in this folder before running the script."
        )

    return pd.read_csv(DATA_PATH)


def print_basic_eda(df: pd.DataFrame) -> None:
    print("\n--- Basic Exploratory Data Analysis ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nColumn Names:")
    print(list(df.columns))
    print("\nMissing Values:")
    print(df.isnull().sum())

    fraud_count = int((df[TARGET_COLUMN] == 1).sum())
    genuine_count = int((df[TARGET_COLUMN] == 0).sum())
    total_count = len(df)

    print(f"\nNumber of Fraudulent Transactions: {fraud_count}")
    print(f"Number of Genuine Transactions: {genuine_count}")
    print("\nClass Imbalance Analysis:")
    print(f"Fraudulent Transactions: {fraud_count / total_count:.4%}")
    print(f"Genuine Transactions: {genuine_count / total_count:.4%}")


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), feature_columns),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    df = load_dataset()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    print_basic_eda(df)

    df = df.dropna()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline(feature_columns)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Model Evaluation ---")
    print(f"Model Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")

    joblib.dump(model, MODEL_PATH)

    feature_metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "feature_defaults": X.median(numeric_only=True).to_dict(),
        "feature_mins": X.min(numeric_only=True).to_dict(),
        "feature_maxs": X.max(numeric_only=True).to_dict(),
    }
    METADATA_PATH.write_text(json.dumps(feature_metadata, indent=2), encoding="utf-8")

    print(f"\nSaved trained model pipeline to: {MODEL_PATH}")
    print(f"Saved feature metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
