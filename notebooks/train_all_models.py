"""
notebooks/train_all_models.py
─────────────────────────────────────────────────────────────────────────────
Full ML pipeline for the RTA Severity Predictor.
Trains ALL 12 models from university syllabus Units I–V:
  Unit II  : Ridge, Lasso (regression → thresholded classification)
  Unit III : Logistic Regression, KNN, Naïve Bayes, SVM, Decision Tree,
             Random Forest, Gradient Boosting, XGBoost, LightGBM
  Unit IV  : MLP Neural Network
  Unit V   : K-Means (visualisation), Optuna tuning of XGBoost (production model)

Preprocessing (Unit I):
  - Drop rows with null target
  - Mode-impute feature nulls
  - LabelEncoder per feature
  - StandardScaler (fit on train only)
  - SMOTE for class imbalance
  - PCA toggle (optional, 95% variance)

Saves all artifacts to app/ml/artifacts/ and writes metrics_report.json.

Usage:
    python notebooks/train_all_models.py
    python notebooks/train_all_models.py --pca          # enable PCA
    python notebooks/train_all_models.py --skip-slow    # skip SVM / Optuna
"""

import sys
import os
import time
import json
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "RTA_Dataset.csv"
ARTIFACTS_DIR = ROOT / "app" / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Target and feature lists ───────────────────────────────────────────────────
TARGET = "Accident_severity"
FEATURE_ORDER = [
    "Day_of_week", "Age_band_of_driver", "Sex_of_driver", "Educational_level",
    "Vehicle_driver_relation", "Driving_experience", "Type_of_vehicle",
    "Owner_of_vehicle", "Service_year_of_vehicle", "Defect_of_vehicle",
    "Area_accident_occured", "Lanes_or_Medians", "Road_allignment",
    "Types_of_Junction", "Road_surface_type", "Road_surface_conditions",
    "Light_conditions", "Weather_conditions", "Type_of_collision",
    "Number_of_vehicles_involved", "Number_of_casualties", "Vehicle_movement",
    "Casualty_class", "Sex_of_casualty", "Age_band_of_casualty",
    "Casualty_severity", "Work_of_casuality", "Fitness_of_casuality",
    "Pedestrian_movement", "Cause_of_accident", "Hour_of_day",
]

SEVERITY_MAP = {
    "Slight Injury": 0,
    "Serious Injury": 1,
    "Fatal injury": 2,
}

CV_FOLDS = 5
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
def banner(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Load and clean data
# ──────────────────────────────────────────────────────────────────────────────
def load_and_clean(path: Path) -> pd.DataFrame:
    banner("Phase 1 — Loading & Cleaning Data")
    logger.info("Reading: %s", path)
    df = pd.read_csv(path)
    logger.info("Shape: %s", df.shape)

    # Drop rows with null target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    logger.info("Dropped %d rows with null target", before - len(df))

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Keep only needed columns
    cols_needed = FEATURE_ORDER + [TARGET]
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset missing columns: {missing_cols}\n"
            f"Available: {df.columns.tolist()}"
        )
    df = df[cols_needed].copy()

    # Mode-impute feature nulls (Unit I)
    for col in FEATURE_ORDER:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info("  Imputed %d nulls in '%s' with mode '%s'", null_count, col, mode_val)

    # Map target to integer codes
    df[TARGET] = df[TARGET].map(SEVERITY_MAP)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    class_counts = df[TARGET].value_counts().sort_index()
    for code, label in [(0, "Slight"), (1, "Serious"), (2, "Fatal")]:
        n = class_counts.get(code, 0)
        pct = n / len(df) * 100
        logger.info("  Class %d (%s): %d (%.1f%%)", code, label, n, pct)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Encode + Scale + SMOTE
# ──────────────────────────────────────────────────────────────────────────────
def encode_and_split(df: pd.DataFrame, use_pca: bool = False):
    banner("Phase 2 — Encoding, Scaling, SMOTE")
    X = df[FEATURE_ORDER].copy()
    y = df[TARGET].values

    # LabelEncode every feature
    encoders = {}
    for col in FEATURE_ORDER:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col.lower()] = le  # save with lowercase key to match form fields

    X = X.values.astype(float)

    # Train/test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    # StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE on training set
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE — Train: %d  (was %d)", len(X_train_sm), len(X_train)
    )
    from collections import Counter
    logger.info("SMOTE class distribution: %s", dict(Counter(y_train_sm)))

    # Optional PCA (toggle flag)
    if use_pca:
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_train_sm = pca.fit_transform(X_train_sm)
        X_test = pca.transform(X_test)
        logger.info("PCA: %d components (95%% variance)", pca.n_components_)
        joblib.dump(pca, ARTIFACTS_DIR / "pca.pkl")

    # Save preprocessing artifacts
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.pkl")
    logger.info("Saved: scaler.pkl, encoders.pkl")

    return X_train_sm, X_test, y_train_sm, y_test, scaler, encoders


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Helpers
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(model, X_test, y_test, train_time: float) -> dict:
    """Compute all classifier metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "train_time_seconds": round(train_time, 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Slight", "Serious", "Fatal"],
            output_dict=True, zero_division=0
        ),
    }

    # ROC-AUC (needs predict_proba)
    try:
        y_prob = model.predict_proba(X_test)
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        )
    except Exception:
        metrics["roc_auc"] = 0.0

    return metrics


def evaluate_regressor(model, X_test, y_test, train_time: float) -> dict:
    """Threshold regression output to 3 classes, compute same metrics."""
    y_raw = model.predict(X_test)
    y_pred = np.where(y_raw < 0.5, 0, np.where(y_raw < 1.5, 1, 2))
    y_pred = np.clip(y_pred.astype(int), 0, 2)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": 0.0,
        "mse": float(mean_squared_error(y_test, y_raw)),
        "mae": float(mean_absolute_error(y_test, y_raw)),
        "r2": float(r2_score(y_test, y_raw)),
        "train_time_seconds": round(train_time, 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Slight", "Serious", "Fatal"],
            output_dict=True, zero_division=0
        ),
    }


def cv_score(model, X, y, label: str) -> float:
    """StratifiedKFold weighted-F1 cross-validation score."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        model, X, y, cv=skf, scoring="f1_weighted", return_train_score=False, n_jobs=-1
    )
    mean_f1 = float(np.mean(results["test_score"]))
    std_f1 = float(np.std(results["test_score"]))
    logger.info("  CV (5-fold) weighted F1: %.4f ± %.4f", mean_f1, std_f1)
    return mean_f1


def train_and_save(key: str, model, X_train, X_test, y_train, y_test,
                   label: str, is_regression: bool = False) -> dict:
    """Train, evaluate, cross-validate, and save one model."""
    logger.info("Training: %s", label)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    if is_regression:
        m = evaluate_regressor(model, X_test, y_test, train_time)
    else:
        m = evaluate_classifier(model, X_test, y_test, train_time)
        _ = cv_score(model, X_train, y_train, label)  # log CV score

    logger.info(
        "  Accuracy: %.4f  Weighted F1: %.4f  ROC-AUC: %.4f  Time: %s",
        m["accuracy"], m["weighted_f1"], m.get("roc_auc", 0.0), fmt_time(train_time)
    )

    out_path = ARTIFACTS_DIR / f"model_{key}.pkl"
    joblib.dump(model, out_path)
    logger.info("  Saved: %s", out_path.name)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Train all models
# ──────────────────────────────────────────────────────────────────────────────
def train_all_models(
    X_train, X_test, y_train, y_test, skip_slow: bool = False
) -> dict:
    metrics_report = {}

    # ── Unit II: Regression baselines ─────────────────────────────────────────
    banner("Unit II — Regression Baselines")

    metrics_report["ridge"] = train_and_save(
        "ridge", Ridge(alpha=1.0), X_train, X_test, y_train, y_test,
        "Ridge Regression", is_regression=True
    )
    metrics_report["lasso"] = train_and_save(
        "lasso", Lasso(alpha=0.01, max_iter=10000), X_train, X_test, y_train, y_test,
        "Lasso Regression", is_regression=True
    )

    # ── Unit III: Classifiers ──────────────────────────────────────────────────
    banner("Unit III — Classifiers")

    metrics_report["lr"] = train_and_save(
        "lr",
        LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test, "Logistic Regression"
    )

    metrics_report["knn"] = train_and_save(
        "knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
        X_train, X_test, y_train, y_test, "k-Nearest Neighbors"
    )

    metrics_report["nb"] = train_and_save(
        "nb", GaussianNB(),
        X_train, X_test, y_train, y_test, "Naïve Bayes"
    )

    if not skip_slow:
        metrics_report["svm"] = train_and_save(
            "svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
            X_train, X_test, y_train, y_test, "Support Vector Machine"
        )
    else:
        logger.info("Skipping SVM (--skip-slow flag)")

    metrics_report["dt"] = train_and_save(
        "dt", DecisionTreeClassifier(max_depth=10, criterion="gini", random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test, "Decision Tree"
    )

    metrics_report["rf"] = train_and_save(
        "rf", RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test, "Random Forest"
    )

    metrics_report["gb"] = train_and_save(
        "gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test, "Gradient Boosting"
    )

    metrics_report["xgb"] = train_and_save(
        "xgb",
        xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        X_train, X_test, y_train, y_test, "XGBoost"
    )

    metrics_report["lgbm"] = train_and_save(
        "lgbm",
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        X_train, X_test, y_train, y_test, "LightGBM"
    )

    # ── Unit IV: MLP Neural Network ────────────────────────────────────────────
    banner("Unit IV — MLP Neural Network")

    metrics_report["mlp"] = train_and_save(
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
            max_iter=500, early_stopping=True, random_state=RANDOM_STATE
        ),
        X_train, X_test, y_train, y_test, "MLP Neural Network"
    )

    return metrics_report


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Unit V: K-Means + Optuna XGBoost tuning
# ──────────────────────────────────────────────────────────────────────────────
def kmeans_analysis(X_train: np.ndarray, y_train: np.ndarray) -> None:
    banner("Unit V — K-Means Clustering")
    logger.info("Fitting K-Means (k=3) on training features...")
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    clusters = km.fit_predict(X_train)

    # Cluster vs true label overlap
    from collections import Counter
    for c in range(3):
        mask = clusters == c
        dist = Counter(y_train[mask])
        logger.info(
            "  Cluster %d (n=%d): Slight=%d, Serious=%d, Fatal=%d",
            c, mask.sum(), dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)
        )
    joblib.dump(km, ARTIFACTS_DIR / "kmeans.pkl")
    logger.info("Saved: kmeans.pkl")


def optuna_tune_xgb(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    n_trials: int = 100
) -> dict:
    banner("Unit V — Optuna XGBoost Hyperparameter Tuning (100 trials)")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=1)
        return float(np.mean(scores["test_score"]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best params: %s", study.best_params)
    logger.info("Best CV F1: %.4f", study.best_value)

    # Retrain best model on full training set
    best_params = {**study.best_params, "use_label_encoder": False,
                   "eval_metric": "mlogloss", "random_state": RANDOM_STATE, "n_jobs": -1}
    best_xgb = xgb.XGBClassifier(**best_params)
    t0 = time.time()
    best_xgb.fit(X_train, y_train)
    train_time = time.time() - t0

    m = evaluate_classifier(best_xgb, X_test, y_test, train_time)
    logger.info(
        "Tuned XGBoost — Accuracy: %.4f  Weighted F1: %.4f",
        m["accuracy"], m["weighted_f1"]
    )

    # Overwrite xgb artifact with tuned version
    joblib.dump(best_xgb, ARTIFACTS_DIR / "model_xgb.pkl")
    logger.info("Replaced model_xgb.pkl with Optuna-tuned version")

    # Save feature importance to metrics
    imp = best_xgb.feature_importances_
    from app.ml.features import FEATURE_ORDER as FO, FEATURE_DISPLAY
    feat_importance = {
        FEATURE_DISPLAY.get(FO[i], FO[i]): round(float(imp[i]), 5)
        for i in range(len(FO))
    }
    feat_importance = dict(
        sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    )
    m["feature_importance"] = feat_importance
    m["optuna_best_params"] = study.best_params

    return m


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6 — SHAP explainer
# ──────────────────────────────────────────────────────────────────────────────
def build_shap_explainer(X_train: np.ndarray) -> None:
    banner("SHAP — Building TreeExplainer")
    try:
        import shap
        xgb_model = joblib.load(ARTIFACTS_DIR / "model_xgb.pkl")
        explainer = shap.TreeExplainer(xgb_model)
        # Warm-up on small sample to catch errors early
        _ = explainer.shap_values(X_train[:50])
        joblib.dump(explainer, ARTIFACTS_DIR / "shap_explainer.pkl")
        logger.info("Saved: shap_explainer.pkl")
    except Exception as exc:
        logger.warning("Could not build SHAP explainer: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Print summary table
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(metrics_report: dict) -> None:
    banner("Summary — All Models ranked by Weighted F1")
    ranked = sorted(
        metrics_report.items(),
        key=lambda x: x[1].get("weighted_f1", 0),
        reverse=True,
    )

    header = f"{'Rank':<5} {'Model':<22} {'Accuracy':>9} {'Wt F1':>8} {'Macro F1':>9} {'ROC-AUC':>8} {'Time':>8}"
    print(header)
    print("─" * len(header))
    for i, (key, m) in enumerate(ranked, 1):
        print(
            f"{i:<5} {key:<22} "
            f"{m['accuracy'] * 100:>8.2f}% "
            f"{m['weighted_f1']:>8.4f} "
            f"{m['macro_f1']:>9.4f} "
            f"{m.get('roc_auc', 0):>8.4f} "
            f"{fmt_time(m['train_time_seconds']):>8}"
        )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train all RTA severity models.")
    parser.add_argument("--pca", action="store_true", help="Apply PCA (95% variance)")
    parser.add_argument("--skip-slow", action="store_true", help="Skip SVM + Optuna")
    args = parser.parse_args()

    total_start = time.time()
    banner("RTA Severity Predictor — Full Training Pipeline")
    logger.info("Data path: %s", DATA_PATH)
    logger.info("Artifacts: %s", ARTIFACTS_DIR)

    if not DATA_PATH.exists():
        print(
            f"\n❌  Dataset not found: {DATA_PATH}\n"
            "    Download from Kaggle and place at data/RTA_Dataset.csv\n"
            "    https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents"
        )
        sys.exit(1)

    # Phase 1
    df = load_and_clean(DATA_PATH)

    # Phase 2
    X_train, X_test, y_train, y_test, scaler, encoders = encode_and_split(df, use_pca=args.pca)

    # Phase 3+4
    metrics_report = train_all_models(X_train, X_test, y_train, y_test, skip_slow=args.skip_slow)

    # Phase 5
    kmeans_analysis(X_train, y_train)

    if not args.skip_slow:
        tuned_metrics = optuna_tune_xgb(X_train, X_test, y_train, y_test, n_trials=100)
        metrics_report["xgb"] = tuned_metrics  # replace with tuned version
    else:
        logger.info("Skipping Optuna tuning (--skip-slow flag)")

    # Phase 6
    build_shap_explainer(X_train)

    # Save metrics report
    report_path = ARTIFACTS_DIR / "metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info("Saved: metrics_report.json")

    # Phase 7
    print_summary(metrics_report)

    total_time = time.time() - total_start
    print(f"\n✅  All done in {fmt_time(total_time)}")
    print(f"    Artifacts saved to: {ARTIFACTS_DIR}\n")


if __name__ == "__main__":
    main()
