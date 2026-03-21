"""
Predictive modeling for Chicago crime.

This module implements three classification tasks:
    1. Crime Category: 5 classes (Violent, Theft, Property, Narcotics, Other)
    2. Beat: Top 25 beats by crime count (location deployment target)
    3. HourBin2: 2 bins (Night 0-12, Day 12-24) for time-of-day prediction

Uses Logistic Regression, Random Forest, Gradient Boosting with GridSearchCV
for Tasks 1 and 2; GB, ExtraTrees, SVM_linear for Task 3 (HourBin2). All
experiments are logged to EXPERIMENT_LOG for comparison.

Source: Extracted from notebooks/04_data_modeling.ipynb
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from .load_data import DATA_PROCESSED
from .preprocess import load_cleaned

# ------------------------------------------------------------------------------
# Experiment logging
# ------------------------------------------------------------------------------
# Mutable list; each call to log_experiment() appends one record.
# Reset at the start of run_full_pipeline().
EXPERIMENT_LOG: list[dict[str, Any]] = []


def log_experiment(
    task: str,
    model_name: str,
    params: dict | Any,
    accuracy: float,
    f1: float,
    notes: str = "",
) -> None:
    """
    Append one experiment result to EXPERIMENT_LOG.

    Args:
        task: Task name (e.g., 'Crime_Category', 'Beat', 'HourBin2').
        model_name: Model identifier (e.g., 'LR', 'GB_tuned', 'SVM_linear').
        params: Hyperparameters or config (converted to string for storage).
        accuracy: Test accuracy (0-1).
        f1: Weighted F1 score (0-1).
        notes: Optional note (e.g., 'GridSearch', '2 bins').
    """
    EXPERIMENT_LOG.append(
        {
            "task": task,
            "model": model_name,
            "params": str(params),
            "accuracy": round(float(accuracy), 4),
            "f1_weighted": round(float(f1), 4),
            "notes": notes,
        }
    )


# ------------------------------------------------------------------------------
# Data preparation for modeling
# ------------------------------------------------------------------------------

def ensure_modeling_columns(df: pd.DataFrame, sample_size: int = 200000) -> pd.DataFrame:
    """
    Add modeling-specific columns if missing (e.g., for older cleaned datasets).

    Adds: Lat_bin, Lon_bin, HourBin, Crime_Category (if not present),
    Location_enc (top 20 locations, label-encoded). Optionally samples
    to sample_size for faster training.

    Args:
        df: Cleaned DataFrame (from load_cleaned or preprocess output).
        sample_size: If > 0 and len(df) > sample_size, random sample is taken.
                     Set to None to use full dataset.

    Returns:
        DataFrame with all required columns; possibly sampled.
    """
    df = df.copy()
    if "Lat_bin" not in df.columns:
        df["Lat_bin"] = (df["Latitude"] // 0.02).astype(int)
    if "Lon_bin" not in df.columns:
        df["Lon_bin"] = (df["Longitude"] // 0.02).astype(int)
    if "HourBin" not in df.columns:
        # 6-bin definition matching preprocess.get_hour_bin
        def get_hour_bin(h: int) -> int:
            if 0 <= h < 4: return 0
            if 4 <= h < 8: return 1
            if 8 <= h < 12: return 2
            if 12 <= h < 16: return 3
            if 16 <= h < 20: return 4
            return 5
        df["HourBin"] = df["Hour"].apply(get_hour_bin)
    if "Crime_Category" not in df.columns:
        CRIME_MAP = {
            "Violent": ["BATTERY", "ASSAULT", "ROBBERY", "CRIMINAL SEXUAL ASSAULT", "HOMICIDE", "KIDNAPPING"],
            "Theft": ["THEFT", "MOTOR VEHICLE THEFT", "BURGLARY"],
            "Property": ["CRIMINAL DAMAGE", "CRIMINAL TRESPASS", "ARSON"],
            "Narcotics": ["NARCOTICS", "OTHER NARCOTIC VIOLATION"],
            "Other": [],
        }
        def map_cat(pt: str) -> str:
            for c, t in CRIME_MAP.items():
                if pt in t: return c
            return "Other"
        df["Crime_Category"] = df["Primary Type"].apply(map_cat)

    # Location_enc: top 20 locations, others mapped to 'OTHER', then label-encoded
    top_loc = df["Location Description"].value_counts().head(20).index.tolist()
    df["Location_enc"] = df["Location Description"].apply(lambda x: x if x in top_loc else "OTHER")
    df["Location_enc"] = LabelEncoder().fit_transform(df["Location_enc"].astype(str))

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df


def prepare_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build feature matrices and targets for all three tasks.

    Task 1 (Crime Category): feat_crime = Beat, Lat_bin, Lon_bin, HourBin, etc.
    Task 2 (Beat): Restrict to top 25 beats; feat_beat2 includes Crime_Cat_enc.
    Task 3 (HourBin2): 2 bins (Night 0-12, Day 12-24); feat_hour_alt includes
                       Beat, Crime_Cat_enc, spatial and temporal features.

    Returns:
        Dict with keys: X1, y1, X2, y2, X3_alt, y3_2bin, feat_crime, feat_beat2, feat_hour_alt.
    """
    df["Community_Area"] = df["Community Area"].fillna(-1).astype(int)
    df["Crime_Cat_enc"] = LabelEncoder().fit_transform(df["Crime_Category"])

    # Task 1: Crime Category (5 classes)
    feat_crime = ["Beat", "Lat_bin", "Lon_bin", "HourBin", "DayOfWeek", "Month", "Location_enc", "Domestic"]
    X1 = df[feat_crime].astype(float)
    y1 = df["Crime_Category"]

    # Task 2: Beat (top 25 for class balance; predicts which beat has the crime)
    top_beats = df["Beat"].value_counts().head(25).index.tolist()
    mask_b = df["Beat"].isin(top_beats)
    feat_beat2 = ["Lat_bin", "Lon_bin", "Crime_Cat_enc", "HourBin", "DayOfWeek", "Month", "Location_enc", "Community_Area"]
    X2 = df.loc[mask_b, feat_beat2].astype(float)
    y2 = df.loc[mask_b, "Beat"].astype(str)

    # Task 3: HourBin2 (2 bins - Night=0, Day=1; random baseline 50%)
    def hour_to_2bin(h: int) -> int:
        return 0 if 0 <= h < 12 else 1
    df["HourBin2"] = df["Hour"].apply(hour_to_2bin)
    feat_hour_alt = [
        "Beat", "Crime_Cat_enc", "DayOfWeek", "Month", "Location_enc",
        "IsWeekend", "Lat_bin", "Lon_bin", "Community_Area",
    ]
    X3_alt = df[feat_hour_alt].copy()
    X3_alt["IsWeekend"] = X3_alt["IsWeekend"].astype(int)
    X3_alt = X3_alt.astype(float)
    y3_2bin = df["HourBin2"]

    return {
        "X1": X1, "y1": y1,
        "X2": X2, "y2": y2,
        "X3_alt": X3_alt, "y3_2bin": y3_2bin,
        "feat_crime": feat_crime, "feat_beat2": feat_beat2, "feat_hour_alt": feat_hour_alt,
    }


# ------------------------------------------------------------------------------
# Model training and evaluation
# ------------------------------------------------------------------------------

def run_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    task_name: str,
    min_samples: int = 2,
) -> tuple[Any, np.ndarray, pd.Series, StandardScaler]:
    """
    Run LR, RF, GB baseline models + GridSearchCV on GB for one task.

    Filters to classes with >= min_samples (avoids stratify errors).
    80/20 train/test split, StandardScaler fit on train only.
    Best model is the GridSearch result (GB_tuned).

    Args:
        X: Feature matrix.
        y: Target (labels).
        task_name: For logging (e.g., 'Crime_Category', 'Beat').
        min_samples: Min samples per class to keep.

    Returns:
        (gs, X_te, y_te, scaler) where gs is the fitted GridSearchCV object.
    """
    class_counts = y.value_counts()
    valid = class_counts[class_counts >= min_samples].index
    m = y.isin(valid)
    Xf = X[m].reset_index(drop=True)
    yf = y[m].reset_index(drop=True)

    X_tr, X_te, y_tr, y_te = train_test_split(Xf, yf, test_size=0.2, random_state=42, stratify=yf)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Baseline models
    models = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    for name, mdl in models.items():
        mdl.fit(X_tr_s, y_tr)
        pred = mdl.predict(X_te_s)
        log_experiment(task_name, name, {}, accuracy_score(y_te, pred), f1_score(y_te, pred, average="weighted"))

    # GridSearch on GB (best baseline typically)
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 8, 10], "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0,
    )
    gs.fit(X_tr_s, y_tr)
    pred_gs = gs.predict(X_te_s)
    log_experiment(task_name, "GB_tuned", gs.best_params_, accuracy_score(y_te, pred_gs), f1_score(y_te, pred_gs, average="weighted"), "GridSearch")

    return gs, X_te_s, y_te, scaler


def run_hourbin2_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """
    Run GB, ExtraTrees, SVM_linear on HourBin2 (2-class task).

    Uses alternate feature set (feat_hour_alt) and simpler models than
    Tasks 1/2. Returns predictions from the best-performing model by accuracy.

    Returns:
        (X_te, y_te, pred_best) for use in classification reports.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    best_acc, pred_best = 0, None
    for name, mdl in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("SVM_linear", LinearSVC(max_iter=2000, random_state=42, dual="auto")),
    ]:
        mdl.fit(X_tr_s, y_tr)
        pred = mdl.predict(X_te_s)
        acc = accuracy_score(y_te, pred)
        log_experiment("HourBin2", name, {}, acc, f1_score(y_te, pred, average="weighted"), "2 bins")
        if acc > best_acc:
            best_acc, pred_best = acc, pred
    return X_te_s, y_te, pred_best


def run_full_pipeline(
    df: pd.DataFrame | None = None,
    filepath: Path | str | None = None,
    sample_size: int = 200000,
) -> dict[str, Any]:
    """
    Execute the complete modeling pipeline: load, prepare, train, evaluate.

    Runs all three tasks and populates EXPERIMENT_LOG. Use the returned
    dict to call print_classification_reports() or plot_feature_importance().

    Args:
        df: Optional DataFrame. If None, loads from filepath or default.
        filepath: Path to cleaned CSV when df is None.
        sample_size: Passed to ensure_modeling_columns for sampling.

    Returns:
        Dict with gs1, X1_te, y1_te, gs2, X2_te, y2_te, X3_te, y3_te, pred3,
        experiment_log, feat_crime.
    """
    global EXPERIMENT_LOG
    EXPERIMENT_LOG = []

    if df is None:
        df = load_cleaned(filepath or (DATA_PROCESSED / "crimes_cleaned.csv"))
    df = ensure_modeling_columns(df, sample_size=sample_size)
    data = prepare_features(df)

    # Task 1: Crime Category
    gs1, X1_te, y1_te, scaler1 = run_experiments(data["X1"], data["y1"], "Crime_Category")
    # Task 2: Beat
    gs2, X2_te, y2_te, scaler2 = run_experiments(data["X2"], data["y2"], "Beat")
    # Task 3: HourBin2
    X3_te, y3_te, pred3 = run_hourbin2_models(data["X3_alt"], data["y3_2bin"])

    return {
        "gs1": gs1, "X1_te": X1_te, "y1_te": y1_te,
        "gs2": gs2, "X2_te": X2_te, "y2_te": y2_te,
        "X3_te": X3_te, "y3_te": y3_te, "pred3": pred3,
        "experiment_log": EXPERIMENT_LOG,
        "feat_crime": data["feat_crime"],
    }


# ------------------------------------------------------------------------------
# Reporting utilities
# ------------------------------------------------------------------------------

def print_classification_reports(
    gs1: Any, X1_te: np.ndarray, y1_te: pd.Series,
    gs2: Any, X2_te: np.ndarray, y2_te: pd.Series,
    pred3: np.ndarray, y3_te: pd.Series,
) -> None:
    """
    Print sklearn classification_report for all three tasks.

    gs1, gs2: Fitted GridSearchCV (best model) for Crime Category and Beat.
    pred3: Best HourBin2 model predictions (from run_hourbin2_models).
    """
    print("=== Crime Category - Best Model ===")
    pred1 = gs1.predict(X1_te)
    print("Accuracy:", round(float(accuracy_score(y1_te, pred1)), 4))
    print(classification_report(y1_te, pred1, zero_division=0))

    print("=== Beat - Best Model ===")
    pred2 = gs2.predict(X2_te)
    print("Accuracy:", round(float(accuracy_score(y2_te, pred2)), 4))
    print(classification_report(y2_te, pred2, zero_division=0))

    print("=== HourBin2 - Best Model (2 bins: Night=0, Day=1) ===")
    print("Accuracy:", round(float(accuracy_score(y3_te, pred3)), 4))
    print(classification_report(y3_te, pred3, target_names=["Night", "Day"], zero_division=0))


def plot_feature_importance(
    X1: pd.DataFrame,
    y1: pd.Series,
    feat_crime: list[str],
) -> None:
    """
    Plot Random Forest feature importance for Crime Category task.

    Trains an RF on a filtered/split subset and plots horizontal bar chart
    of feature importance. Useful for understanding which features drive
    crime type prediction.
    """
    import matplotlib.pyplot as plt

    class_counts = y1.value_counts()
    valid = class_counts[class_counts >= 2].index
    m = y1.isin(valid)
    X1f, y1f = X1[m], y1[m]
    X_tr, X_te, y_tr, y_te = train_test_split(X1f, y1f, test_size=0.2, random_state=42, stratify=y1f)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(StandardScaler().fit_transform(X_tr), y_tr)
    imp = pd.DataFrame({"feature": feat_crime, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    imp.plot(x="feature", y="importance", kind="barh", figsize=(8, 5), legend=False)
    plt.title("Feature Importance (Crime Category)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    result = run_full_pipeline(sample_size=200000)
    print_classification_reports(
        result["gs1"], result["X1_te"], result["y1_te"],
        result["gs2"], result["X2_te"], result["y2_te"],
        result["pred3"], result["y3_te"],
    )
    print("\nExperiment log:")
    print(pd.DataFrame(result["experiment_log"]).to_string(index=False))
