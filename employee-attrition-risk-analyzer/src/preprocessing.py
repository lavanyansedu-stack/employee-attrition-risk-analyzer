from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import DATA_PATH, DROP_COLUMNS, TARGET, RANDOM_STATE, TEST_SIZE


def load_data(path=DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download the IBM HR Analytics Employee Attrition dataset and "
            "save it as data/employee_attrition.csv."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove exact duplicate records.
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove only columns defined as identifiers/constants in the project plan.
    existing_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    # Convert target to binary.
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    if df[TARGET].isna().any():
        raise ValueError("Unexpected values found in Attrition. Expected Yes/No.")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 24, 35, 45, 60, np.inf],
            labels=["Under 25", "25-35", "36-45", "46-60", "60+"],
            include_lowest=True,
        ).astype(str)

    if "TotalWorkingYears" in df.columns:
        df["ExperienceGroup"] = pd.cut(
            df["TotalWorkingYears"],
            bins=[-np.inf, 2, 5, 10, 20, np.inf],
            labels=["0-2 years", "3-5 years", "6-10 years", "11-20 years", "20+ years"],
        ).astype(str)

    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def prepare_dataset(path=DATA_PATH):
    raw = load_data(path)
    cleaned = clean_data(raw)
    featured = add_engineered_features(cleaned)
    X, y = split_features_target(featured)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    return raw, featured, X_train, X_test, y_train, y_test, preprocessor
