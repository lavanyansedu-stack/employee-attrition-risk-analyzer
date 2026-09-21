import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from .config import (
    DATA_PATH, MODEL_PATH, METRICS_PATH, FEATURE_INFO_PATH,
    RANDOM_STATE
)
from .preprocessing import prepare_dataset


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    (
        raw, featured, X_train, X_test, y_train, y_test, preprocessor
    ) = prepare_dataset(DATA_PATH)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fitted_models[name] = pipe

    # Business priority: recall first, then F1, then ROC-AUC.
    selected_name = max(
        results,
        key=lambda name: (
            results[name]["recall"],
            results[name]["f1"],
            results[name]["roc_auc"],
        ),
    )

    final_model = fitted_models[selected_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    # Store training feature names for dashboard/form validation.
    feature_info = {
        "selected_model": selected_name,
        "input_columns": X_train.columns.tolist(),
        "dataset_rows": int(len(raw)),
        "dataset_columns": int(raw.shape[1]),
    }
    FEATURE_INFO_PATH.write_text(json.dumps(feature_info, indent=2))

    METRICS_PATH.write_text(json.dumps(results, indent=2))

    print("\nMODEL COMPARISON")
    print("-" * 75)
    for name, m in results.items():
        print(
            f"{name:20s} "
            f"Accuracy={m['accuracy']:.3f}  "
            f"Precision={m['precision']:.3f}  "
            f"Recall={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  "
            f"ROC-AUC={m['roc_auc']:.3f}"
        )

    print(f"\nSelected model: {selected_name}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
