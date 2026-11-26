"""
FinBridge Credit Risk Model Training
Run: python train_model.py

Uses row-wise feature engineering and fold-safe sklearn pipelines.
"""
import os
import re
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
TARGET = "default_flag"
ID_COLUMN = "application_id"


class FinBridgeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create applicant-level features independently for each row."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[ID_COLUMN, "applicant_ref_id"], errors="ignore")

        # Missing-value flags are created before imputing anything.
        original_columns = list(X.columns)
        for col in original_columns:
            X[f"{col}_missing"] = X[col].isna().astype("int8")

        # Convert percentage strings such as '10.71%' into numeric values.
        for col in ["interest_rate_pct", "revolving_utilization_pct"]:
            if col in X.columns:
                X[col] = (
                    X[col].astype(str).str.replace("%", "", regex=False)
                    .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                )
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Convert known numeric columns safely.
        numeric_candidates = [
            "requested_amount", "monthly_emi", "annual_income",
            "debt_to_income_ratio", "delinquencies_24m",
            "credit_inquiries_6m", "active_credit_lines",
            "derogatory_public_records", "revolving_credit_balance",
            "total_credit_accounts", "recent_payment_amount",
            "interest_rate_pct", "revolving_utilization_pct"
        ]
        for col in numeric_candidates:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Convert risk subtier A1..G5 to a single ordered value 1..35.
        if "risk_subtier" in X.columns:
            def subtier_number(value):
                if pd.isna(value):
                    return np.nan
                match = re.fullmatch(r"\s*([A-G])\s*([1-5])\s*", str(value).upper())
                if not match:
                    return np.nan
                letter = ord(match.group(1)) - ord("A")
                digit = int(match.group(2))
                return letter * 5 + digit
            X["risk_subtier_number"] = X["risk_subtier"].map(subtier_number)
            X = X.drop(columns=["risk_subtier"])

        # Safe row-wise ratios. Invalid or zero denominators become missing
        # and are handled by the pipeline's training-fold imputer.
        income = X["annual_income"].replace(0, np.nan) if "annual_income" in X else np.nan

        if "monthly_emi" in X and "annual_income" in X:
            X["yearly_emi_to_annual_income"] = (X["monthly_emi"] * 12) / income

        if "requested_amount" in X and "annual_income" in X:
            X["requested_amount_to_annual_income"] = X["requested_amount"] / income

        if "revolving_credit_balance" in X and "annual_income" in X:
            X["revolving_balance_to_annual_income"] = X["revolving_credit_balance"] / income

        if "revolving_credit_balance" in X and "total_credit_accounts" in X:
            accounts = X["total_credit_accounts"].replace(0, np.nan)
            X["revolving_balance_per_credit_account"] = (
                X["revolving_credit_balance"] / accounts
            )

        # Combined count of recent delinquencies and hard inquiries.
        if "delinquencies_24m" in X and "credit_inquiries_6m" in X:
            X["delinquency_inquiry_total"] = (
                X["delinquencies_24m"].fillna(0)
                + X["credit_inquiries_6m"].fillna(0)
            )

        # Make all numeric infinities missing for safe imputation.
        X = X.replace([np.inf, -np.inf], np.nan)
        return X


def build_preprocessor(X, dense=True):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=not dense
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ], remainder="drop")


def make_models(X_sample):
    engineered = FinBridgeFeatureEngineer().fit_transform(X_sample)
    # Dense one-hot output is required by HistGradientBoostingClassifier.
    preprocessor = build_preprocessor(engineered, dense=True)

    logistic = Pipeline([
        ("features", FinBridgeFeatureEngineer()),
        ("preprocess", clone(preprocessor)),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE
        ))
    ])

    hgb = Pipeline([
        ("features", FinBridgeFeatureEngineer()),
        ("preprocess", clone(preprocessor)),
        ("model", HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE
        ))
    ])
    return logistic, hgb


def rank_scale(train_scores, new_scores):
    """Scale new scores to [0,1] using the training score distribution."""
    sorted_scores = np.sort(np.asarray(train_scores))
    return np.searchsorted(sorted_scores, np.asarray(new_scores), side="right") / len(sorted_scores)


def best_f1_threshold(y_true, probabilities):
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, probabilities >= t, zero_division=0)
        for t in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def metric_row(name, y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    return {
        "model": name,
        "ROC-AUC": roc_auc_score(y_true, scores),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "threshold": threshold
    }


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    if TARGET not in train.columns:
        raise ValueError("train.csv must contain the default_flag target column.")

    y = train[TARGET].astype(int)
    X = train.drop(columns=[TARGET])

    # Holdout validation is used for threshold selection and permutation
    # importance. The five-fold CV below is performed only on development data.
    X_dev, X_val, y_dev, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    logistic, hgb = make_models(X_dev)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n1) Five-fold ROC-AUC comparison")
    log_cv = cross_val_score(
        logistic, X_dev, y_dev, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    print(f"Logistic Regression: {log_cv.mean():.4f} +/- {log_cv.std():.4f}")

    # A deliberately small tuning grid: three quick configurations.
    hgb_options = [
        {"max_iter": 80, "learning_rate": 0.08, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"max_iter": 120, "learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"max_iter": 100, "learning_rate": 0.08, "max_leaf_nodes": 25, "l2_regularization": 2.0},
    ]

    best_hgb_params = None
    best_hgb_auc = -np.inf
    print("\n2) Small HistGradientBoosting tuning")
    for i, params in enumerate(hgb_options, start=1):
        # Apply each model parameter explicitly.
        candidate = clone(hgb)
        candidate.set_params(**{f"model__{k}": v for k, v in params.items()})
        scores = cross_val_score(
            candidate, X_dev, y_dev, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mean_auc = scores.mean()
        print(f"Setting {i}: ROC-AUC {mean_auc:.4f} +/- {scores.std():.4f}")
        if mean_auc > best_hgb_auc:
            best_hgb_auc = mean_auc
            best_hgb_params = params

    hgb.set_params(**{f"model__{k}": v for k, v in best_hgb_params.items()})
    print("Selected gradient boosting settings:", best_hgb_params)

    print("\n3) Generate out-of-fold scores for fair blend comparison")
    log_oof = cross_val_predict(
        logistic, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    hgb_oof = cross_val_predict(
        hgb, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    log_auc = roc_auc_score(y_dev, log_oof)
    hgb_auc = roc_auc_score(y_dev, hgb_oof)

    log_rank = rankdata(log_oof, method="average") / len(log_oof)
    hgb_rank = rankdata(hgb_oof, method="average") / len(hgb_oof)
    blend_oof = (log_rank + hgb_rank) / 2
    blend_auc = roc_auc_score(y_dev, blend_oof)

    print(f"Logistic OOF ROC-AUC: {log_auc:.4f}")
    print(f"HistGradientBoosting OOF ROC-AUC: {hgb_auc:.4f}")
    print(f"Rank blend OOF ROC-AUC: {blend_auc:.4f}")

    if blend_auc > max(log_auc, hgb_auc):
        selected_name = "Rank Blend"
        selected_kind = "blend"
        print("The rank blend beats both individual models.")
    else:
        selected_name = "Logistic Regression" if log_auc >= hgb_auc else "HistGradientBoosting"
        selected_kind = "logistic" if log_auc >= hgb_auc else "hgb"
        print(f"The blend does not beat both models. Using {selected_name}.")

    print("\n4) Fit models and evaluate on untouched validation split")
    logistic.fit(X_dev, y_dev)
    hgb.fit(X_dev, y_dev)

    log_val_raw = logistic.predict_proba(X_val)[:, 1]
    hgb_val_raw = hgb.predict_proba(X_val)[:, 1]
    log_val = rank_scale(log_oof, log_val_raw)
    hgb_val = rank_scale(hgb_oof, hgb_val_raw)
    blend_val = (log_val + hgb_val) / 2

    # Use rank-scaled validation scores for all models so the blend is comparable.
    scores_for_validation = {
        "Logistic Regression": log_val,
        "HistGradientBoosting": hgb_val,
        "Rank Blend": blend_val
    }

    results = []
    thresholds = {}
    for name, scores in scores_for_validation.items():
        threshold = best_f1_threshold(y_val, scores)
        thresholds[name] = threshold
        results.append(metric_row(name, y_val, scores, threshold))

    results_df = pd.DataFrame(results)
    print("\nValidation metrics (threshold chosen to maximize validation F1):")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    selected_threshold = thresholds[selected_name]
    if selected_kind == "logistic":
        validation_scores = log_val
    elif selected_kind == "hgb":
        validation_scores = hgb_val
    else:
        validation_scores = blend_val

    print("\n5) Permutation importance on validation data")
    # Permutation importance is calculated on the selected model. For a blend,
    # report both constituent models' importances and average their rankings.
    if selected_kind == "blend":
        importance_log = permutation_importance(
            logistic, X_val, y_val, scoring="roc_auc",
            n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
        )
        importance_hgb = permutation_importance(
            hgb, X_val, y_val, scoring="roc_auc",
            n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
        )
        # Feature importance is computed on original input columns.
        imp_values = (importance_log.importances_mean + importance_hgb.importances_mean) / 2
    else:
        selected_model = logistic if selected_kind == "logistic" else hgb
        importance = permutation_importance(
            selected_model, X_val, y_val, scoring="roc_auc",
            n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
        )
        imp_values = importance.importances_mean

    importance_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance": imp_values
    }).sort_values("importance", ascending=False)
    print(importance_df.head(10).to_string(index=False))

    # Refit selected model(s) on all labelled rows for final test predictions.
    print("\n6) Refit on all training rows and create submission.csv")
    if selected_kind == "logistic":
        logistic.fit(X, y)
        final_model = logistic
        test_scores = final_model.predict_proba(test)[:, 1]
        final_predictions = rank_scale(log_oof, test_scores)
    elif selected_kind == "hgb":
        hgb.fit(X, y)
        final_model = hgb
        test_scores = final_model.predict_proba(test)[:, 1]
        final_predictions = rank_scale(hgb_oof, test_scores)
    else:
        logistic.fit(X, y)
        hgb.fit(X, y)
        final_model = {"logistic": logistic, "hgb": hgb}
        test_log = rank_scale(log_oof, logistic.predict_proba(test)[:, 1])
        test_hgb = rank_scale(hgb_oof, hgb.predict_proba(test)[:, 1])
        final_predictions = (test_log + test_hgb) / 2

    # Keep predictions strictly inside (0,1), as requested by the assignment.
    final_predictions = np.clip(final_predictions, 1e-6, 1 - 1e-6)
    submission = pd.DataFrame({
        "application_id": test[ID_COLUMN],
        "default_flag": final_predictions
    })
    assert list(submission.columns) == ["application_id", "default_flag"]
    assert len(submission) == len(test)
    assert submission["default_flag"].notna().all()
    assert submission["default_flag"].between(0, 1).all()
    submission.to_csv("outputs/submission.csv", index=False)

    results_df.to_csv("outputs/validation_metrics.csv", index=False)
    importance_df.to_csv("outputs/feature_importance.csv", index=False)

    joblib.dump({
        "selected_kind": selected_kind,
        "selected_name": selected_name,
        "selected_threshold": selected_threshold,
        "thresholds": thresholds,
        "logistic_model": logistic,
        "hgb_model": hgb,
        "oof_scores": {
            "logistic": log_oof,
            "hgb": hgb_oof
        },
        "cv_auc": {
            "logistic": float(log_auc),
            "hgb": float(hgb_auc),
            "blend": float(blend_auc)
        },
        "validation_metrics": results_df,
        "feature_importance": importance_df
    }, "models/finbridge_models.pkl")

    print("\nCompleted successfully.")
    print("Selected approach:", selected_name)
    print("Files saved:")
    print("- models/finbridge_models.pkl")
    print("- outputs/submission.csv")
    print("- outputs/validation_metrics.csv")
    print("- outputs/feature_importance.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
