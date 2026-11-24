import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RANDOM_STATE = 42

def make_demo_data(n=1200, readings=30, seed=42):
    """Create synthetic data only for testing the software workflow."""
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, 0.12, n)
    base = rng.lognormal(mean=2.1, sigma=0.45, size=n)
    matrix = []
    for i in range(n):
        noise = rng.normal(0, base[i] * 0.12, readings)
        trend = np.linspace(0, rng.normal(0, base[i] * 0.08), readings)
        vals = np.maximum(0, base[i] + noise + trend)
        if labels[i] == 1:
            # Synthetic irregular pattern, not a representation of all theft behaviour.
            vals[rng.integers(0, readings//2):] *= rng.uniform(0.35, 0.65)
            vals[rng.integers(0, readings)] *= rng.uniform(0.1, 0.4)
        matrix.append(vals)
    df = pd.DataFrame(matrix, columns=[f"reading_{j+1:02d}" for j in range(readings)])
    df.insert(0, "consumer_id", [f"DEMO_{i+1:04d}" for i in range(n)])
    df["label"] = labels
    return df

def prepare_features(df):
    """Convert each consumer's selected numeric readings into interpretable summary features."""
    arr = df.to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        med = np.nanmedian(arr, axis=1)
        minv = np.nanmin(arr, axis=1)
        maxv = np.nanmax(arr, axis=1)
        q25 = np.nanpercentile(arr, 25, axis=1)
        q75 = np.nanpercentile(arr, 75, axis=1)
    safe_mean = np.where(np.abs(mean) < 1e-9, np.nan, mean)
    first = arr[:, 0]
    last = arr[:, -1]
    change = (last - first) / np.abs(np.where(np.abs(first) < 1e-9, np.nan, first))
    zero_freq = np.nanmean(np.isclose(arr, 0, equal_nan=False), axis=1)
    diffs = np.diff(arr, axis=1)
    abs_diff_mean = np.nanmean(np.abs(diffs), axis=1) if arr.shape[1] > 1 else np.zeros(len(arr))
    return pd.DataFrame({
        "mean_consumption": mean,
        "median_consumption": med,
        "std_consumption": std,
        "coefficient_variation": std / safe_mean,
        "min_consumption": minv,
        "max_consumption": maxv,
        "range_consumption": maxv - minv,
        "iqr_consumption": q75 - q25,
        "relative_change_first_last": change,
        "zero_reading_frequency": zero_freq,
        "mean_absolute_step_change": abs_diff_mean,
    }, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0)

def _metrics(y_true, y_pred, scores):
    out = {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if len(np.unique(y_true)) == 2:
        out["ROC-AUC"] = roc_auc_score(y_true, scores)
        out["PR-AUC (Average Precision)"] = average_precision_score(y_true, scores)
    else:
        out["ROC-AUC"] = np.nan
        out["PR-AUC (Average Precision)"] = np.nan
    return out

def train_and_evaluate(X, y):
    if y.nunique() != 2:
        raise ValueError("Both normal (0) and positive (1) labels are required.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    p_lr = logistic.predict_proba(X_test)[:, 1]
    p_rf = forest.predict_proba(X_test)[:, 1]

    iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", IsolationForest(n_estimators=250, contamination="auto", random_state=RANDOM_STATE))
    ])
    iso.fit(X_train)
    # IsolationForest decision_function is lower for more anomalous points.
    raw_anomaly = -iso.decision_function(X_test)
    scaler = MinMaxScaler()
    anomaly = scaler.fit_transform(raw_anomaly.reshape(-1, 1)).ravel()

    # Preliminary fusion; compare it with individual models, do not assume it is superior.
    hybrid = 0.7 * p_rf + 0.3 * anomaly
    threshold = 0.5
    yhat_lr = (p_lr >= threshold).astype(int)
    yhat_rf = (p_rf >= threshold).astype(int)
    yhat_hybrid = (hybrid >= threshold).astype(int)

    rows = []
    for name, pred, score in [
        ("Logistic Regression", yhat_lr, p_lr),
        ("Random Forest", yhat_rf, p_rf),
        ("Hybrid (0.7 RF + 0.3 anomaly)", yhat_hybrid, hybrid),
    ]:
        rows.append({"Model": name, **_metrics(y_test, pred, score)})
    metrics = pd.DataFrame(rows).set_index("Model").reset_index()

    # Score all rows using fitted models for the dashboard review queue.
    all_lr = logistic.predict_proba(X)[:, 1]
    all_rf = forest.predict_proba(X)[:, 1]
    iso_all = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", IsolationForest(n_estimators=250, contamination="auto", random_state=RANDOM_STATE))
    ])
    iso_all.fit(X)
    anomaly_all_raw = -iso_all.decision_function(X)
    anomaly_all = MinMaxScaler().fit_transform(anomaly_all_raw.reshape(-1, 1)).ravel()
    scored = pd.DataFrame({
        "logistic_probability": all_lr,
        "random_forest_probability": all_rf,
        "anomaly_score": anomaly_all,
        "hybrid_risk_score": 0.7 * all_rf + 0.3 * anomaly_all,
    }, index=X.index)
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, yhat_rf, labels=[0, 1]),
        "scored": scored,
        "test_indices": X_test.index,
        "y_test": y_test,
    }

def score_unlabelled_data(X):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE))
    ])
    pipe.fit(X)
    raw = -pipe.decision_function(X)
    score = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).ravel()
    return pd.DataFrame({"anomaly_score": score}, index=X.index)
