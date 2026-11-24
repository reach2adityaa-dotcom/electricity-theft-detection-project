import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.pipeline import (
    make_demo_data, prepare_features, train_and_evaluate,
    score_unlabelled_data
)

st.set_page_config(page_title="Electricity Theft & Consumption Analytics", layout="wide")
st.title("Electricity Theft Detection and Abnormal Consumption Analytics Using Smart Meter Data")
st.caption("Software-only research prototype • Alerts are investigation indicators, not proof of theft.")

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Choose input", ["Upload smart-meter CSV", "Load clearly labelled demo data"])
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) if mode == "Upload smart-meter CSV" else None
    st.markdown("---")
    st.caption("Expected format: one row per consumer, one optional ID column, one label column (if available), and numeric consumption-reading columns.")

if mode == "Load clearly labelled demo data":
    raw = make_demo_data()
    st.warning("DEMO MODE: the data and labels are synthetically generated for software demonstration. They are not real utility records and cannot support claims about real-world theft detection.")
else:
    if uploaded is None:
        st.info("Upload a CSV file to begin. Your file is processed in this session and is not written to the project folder.")
        st.stop()
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

if raw.empty:
    st.error("The selected dataset is empty.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(raw.head(10), use_container_width=True)
st.write(f"Rows: **{len(raw):,}**  |  Columns: **{len(raw.columns):,}**")

all_cols = list(raw.columns)
with st.sidebar:
    id_options = ["(row number)"] + all_cols
    id_col = st.selectbox("Consumer ID column", id_options, index=0)
    label_options = ["No label / unsupervised only"] + all_cols
    label_col = st.selectbox("Label column (normal vs theft/suspicious)", label_options, index=0)
    if label_col != "No label / unsupervised only":
        st.caption("For labels, map the positive class to 1 and normal class to 0 in your CSV before training, or select the positive label below.")
        positive_label = st.text_input("Exact positive label value", value="1")
    else:
        positive_label = "1"

exclude = set()
if id_col != "(row number)":
    exclude.add(id_col)
if label_col != "No label / unsupervised only":
    exclude.add(label_col)

numeric_candidates = [c for c in all_cols if c not in exclude and pd.api.types.is_numeric_dtype(raw[c])]
selected_readings = st.sidebar.multiselect("Consumption columns", numeric_candidates, default=numeric_candidates)
if not selected_readings:
    st.error("Select at least one numeric consumption column.")
    st.stop()

ids = raw[id_col].astype(str) if id_col != "(row number)" else pd.Series([f"Consumer_{i+1}" for i in range(len(raw))])
X = prepare_features(raw[selected_readings])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

st.subheader("Consumption analytics")
c1, c2, c3 = st.columns(3)
c1.metric("Consumers / records", f"{len(raw):,}")
c2.metric("Consumption features", f"{X.shape[1]}")
c3.metric("Selected reading fields", f"{len(selected_readings)}")

summary = X[["mean_consumption", "std_consumption", "min_consumption", "max_consumption"]].describe().T.reset_index(names="Feature")
st.dataframe(summary, use_container_width=True)
fig = px.histogram(X, x="mean_consumption", nbins=30, title="Distribution of mean consumption")
st.plotly_chart(fig, use_container_width=True)

if label_col != "No label / unsupervised only":
    y_raw = raw[label_col]
    positive = positive_label
    # Support explicit 0/1 labels and string labels with a user-provided positive value.
    y = (y_raw.astype(str) == str(positive)).astype(int)
    if y.nunique() < 2:
        st.error("The selected label column contains only one class after applying the positive-label value. Check your label mapping.")
        st.stop()
    st.subheader("Model training and evaluation")
    if st.button("Train models and evaluate on a held-out test split", type="primary"):
        with st.spinner("Training baseline models..."):
            try:
                result = train_and_evaluate(X, y)
                st.session_state["model_result"] = result
                st.session_state["scored_X"] = X.copy()
                st.session_state["consumer_ids"] = ids.copy()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()
    result = st.session_state.get("model_result")
    if result is not None:
        st.markdown("#### Held-out test metrics")
        st.caption("Metrics below are calculated from your uploaded labelled dataset and this run's stratified hold-out split. Record the dataset version, split, and random seed when reporting them.")
        st.dataframe(result["metrics"], use_container_width=True)
        st.markdown("#### Confusion matrix — Random Forest")
        cm = result["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual normal", "Actual positive"], columns=["Predicted normal", "Predicted positive"])
        st.dataframe(cm_df, use_container_width=True)
        st.markdown("#### Ranked review candidates")
        scored = result["scored"].copy()
        scored.insert(0, "consumer_id", ids.loc[scored.index].values)
        st.dataframe(scored.sort_values("hybrid_risk_score", ascending=False).head(50), use_container_width=True)
        st.download_button("Download ranked review candidates (CSV)", scored.to_csv(index=False), "ranked_review_candidates.csv", "text/csv")
else:
    st.subheader("Unsupervised anomaly screening")
    st.caption("Without verified labels, the app can rank unusual consumption patterns but cannot evaluate theft-detection accuracy.")
    if st.button("Generate anomaly ranking", type="primary"):
        scored = score_unlabelled_data(X)
        scored.insert(0, "consumer_id", ids.values)
        st.session_state["unsup_scored"] = scored
    scored = st.session_state.get("unsup_scored")
    if scored is not None:
        st.dataframe(scored.sort_values("anomaly_score", ascending=False).head(50), use_container_width=True)
        st.download_button("Download anomaly ranking (CSV)", scored.to_csv(index=False), "anomaly_ranking.csv", "text/csv")

st.markdown("---")
st.caption("Research note: unusual consumption may arise from occupancy changes, appliance use, weather, meter faults, communication gaps, or other legitimate causes. Human review is required.")
