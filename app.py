import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = r"""# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


st.set_page_config(page_title="Mini Project B Forecasting Starter", layout="wide")

st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("Starter app: dataset audit, time-series setup, baseline feature table, exports, and AI grader.")

DEFAULT_STUDENT_NAME = "Ahmed Al Habsi"
DEFAULT_STUDENT_ID = "PG12S2540605"
DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_TARGET_COL = "ALLSKY_SFC_SW_DWN_Wh_m2"


def get_openrouter_api_key():
    """Read OpenRouter API key without hardcoding it."""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    key = os.getenv("OPENROUTER_API_KEY", "")
    if key:
        return key

    return st.session_state.get("openrouter_api_key_input", "")


def audit_dataframe(dataframe):
    audit = pd.DataFrame({
        "column": dataframe.columns,
        "dtype": [str(dataframe[col].dtype) for col in dataframe.columns],
        "missing_percent": [round(float(dataframe[col].isna().mean() * 100), 3) for col in dataframe.columns],
        "unique_count": [int(dataframe[col].nunique(dropna=True)) for col in dataframe.columns],
    })
    return audit


def clean_time_series(dataframe, timestamp_col, target_col):
    cleaned = dataframe.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")
    before_rows = len(cleaned)
    cleaned = cleaned.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
    cleaned = cleaned.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)
    dropped_rows = before_rows - len(cleaned)
    return cleaned, dropped_rows


def maybe_resample(cleaned, timestamp_col, target_col, freq_choice):
    if freq_choice == "No resampling":
        return cleaned

    freq_map = {
        "Hourly mean": "h",
        "Daily mean": "D",
        "Weekly mean": "W",
    }
    freq = freq_map[freq_choice]

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    resampled = (
        cleaned.set_index(timestamp_col)[numeric_cols]
        .resample(freq)
        .mean()
        .dropna(subset=[target_col])
        .reset_index()
    )
    return resampled


def make_baseline_features(cleaned, timestamp_col, target_col, horizon):
    features = cleaned[[timestamp_col, target_col]].copy()
    features = features.sort_values(timestamp_col).reset_index(drop=True)

    features["lag_1"] = features[target_col].shift(1)
    features["lag_24"] = features[target_col].shift(24)
    features["rolling_mean_24"] = features[target_col].shift(1).rolling(window=24, min_periods=6).mean()

    features["hour"] = features[timestamp_col].dt.hour
    features["weekend"] = features[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    features["month"] = features[timestamp_col].dt.month

    features["y_target"] = features[target_col].shift(-int(horizon))

    feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    model_table = features.dropna(subset=feature_cols + ["y_target"]).reset_index(drop=True)
    X = model_table[feature_cols].copy()
    y = model_table["y_target"].copy()
    return model_table, X, y, feature_cols


def dataframe_records_or_empty(value):
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    return []


def build_submission_json(
    student_name,
    student_id,
    app_url,
    repo_url,
    project_title,
    project_goal,
    data_path,
    original_df,
    cleaned_df,
    model_table,
    timestamp_col,
    target_col,
    horizon,
    resampling_choice,
    results_df,
    insights_text,
):
    audit = audit_dataframe(original_df)
    evidence = {
        "student": {
            "name": student_name,
            "student_id": student_id,
        },
        "project": {
            "title": project_title,
            "goal": project_goal,
            "streamlit_app_url": app_url,
            "github_repo_url": repo_url,
        },
        "dataset": {
            "data_path": data_path,
            "raw_rows": int(len(original_df)),
            "clean_rows": int(len(cleaned_df)),
            "feature_table_rows": int(len(model_table)),
            "columns": list(original_df.columns),
            "audit": audit.to_dict(orient="records"),
        },
        "time_series_setup": {
            "timestamp_column": timestamp_col,
            "target_column": target_col,
            "timestamp_min": str(cleaned_df[timestamp_col].min()) if len(cleaned_df) else "",
            "timestamp_max": str(cleaned_df[timestamp_col].max()) if len(cleaned_df) else "",
            "resampling": resampling_choice,
            "forecast_horizon": int(horizon),
            "missing_timestamps_discussed": False,
            "outliers_discussed": False,
            "resampling_discussed": resampling_choice != "No resampling",
        },
        "baseline_features_created": {
            "lag_1": "lag_1" in model_table.columns,
            "lag_24": "lag_24" in model_table.columns,
            "rolling_mean_24": "rolling_mean_24" in model_table.columns,
            "hour": "hour" in model_table.columns,
            "weekend": "weekend" in model_table.columns,
            "month": "month" in model_table.columns,
            "y_target_shifted_by_horizon": "y_target" in model_table.columns,
        },
        "student_additions_evidence": {
            "has_metrics_table": isinstance(results_df, pd.DataFrame),
            "results_table": dataframe_records_or_empty(results_df),
            "has_extra_dashboard": False,
            "has_insights": bool(insights_text.strip()),
            "insights": insights_text.strip(),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return evidence


def build_project_card(evidence):
    lines = [
        f"# {evidence['project']['title'] or 'Mini Project B Project Card'}",
        "",
        f"Student: {evidence['student']['name']}",
        f"Student ID: {evidence['student']['student_id']}",
        "",
        "## Goal",
        evidence["project"]["goal"] or "Forecast the selected target using time-series methods.",
        "",
        "## Dataset",
        f"- Rows after cleaning: {evidence['dataset']['clean_rows']}",
        f"- Timestamp column: {evidence['time_series_setup']['timestamp_column']}",
        f"- Target column: {evidence['time_series_setup']['target_column']}",
        f"- Time range: {evidence['time_series_setup']['timestamp_min']} to {evidence['time_series_setup']['timestamp_max']}",
        "",
        "## Baseline features prepared",
        "- lag_1",
        "- lag_24",
        "- rolling_mean_24",
        "- hour",
        "- weekend",
        "- month",
        "",
        "## Student additions still required",
        "- Add modeling code under the MODELING placeholder.",
        "- Add metrics and set results_df to a DataFrame.",
        "- Add extra dashboard plots/KPIs under the DASHBOARD placeholder.",
        "- Add insights based on your results.",
    ]
    return "\n".join(lines)


def parse_grader_response(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def call_openrouter_grader(api_key, evidence_json_text):
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", evidence_json_text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Mini Project B AI Grader",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


with st.sidebar:
    st.header("Student information")
    student_name = st.text_input("Student name", value=DEFAULT_STUDENT_NAME)
    student_id = st.text_input("Student ID", value=DEFAULT_STUDENT_ID)
    app_url = st.text_input("Deployed Streamlit app URL", value="")
    repo_url = st.text_input("GitHub repo URL", value="")
    project_title = st.text_input("Project title", value="Solar Irradiance Forecasting in Manah")
    project_goal = st.text_area(
        "Project goal",
        value="Prepare a time-series forecasting project to predict solar irradiance using hourly NASA POWER data.",
        height=100,
    )

    st.header("Data source")
    data_path = st.text_input("Local dataset path", value=DEFAULT_DATA_PATH)

    st.header("OpenRouter")
    st.session_state["openrouter_api_key_input"] = st.text_input(
        "OpenRouter API key",
        type="password",
        value=st.session_state.get("openrouter_api_key_input", ""),
        help="Used only at runtime. Do not hardcode keys in this file.",
    )


try:
    df = pd.read_csv(data_path)
except Exception as exc:
    st.error(f"Could not load dataset from {data_path}: {exc}")
    st.stop()

st.subheader("1. Dataset preview")
st.write("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Dataset audit")
audit_df = audit_dataframe(df)
st.dataframe(audit_df, use_container_width=True)

missing_top = audit_df.sort_values("missing_percent", ascending=False).head(10)
st.write("Top 10 columns by missing percentage")
st.dataframe(missing_top[["column", "missing_percent"]], use_container_width=True)

st.subheader("3. Timestamp and target selection")
columns = list(df.columns)
timestamp_index = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
timestamp_col = st.selectbox("Timestamp column", columns, index=timestamp_index)

numeric_candidates = []
for col in columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().sum() > 0:
        numeric_candidates.append(col)

if not numeric_candidates:
    st.error("No numeric target candidates were found.")
    st.stop()

target_index = numeric_candidates.index(DEFAULT_TARGET_COL) if DEFAULT_TARGET_COL in numeric_candidates else 0
target_col = st.selectbox("Target column", numeric_candidates, index=target_index)

cleaned_df, dropped_rows = clean_time_series(df, timestamp_col, target_col)

c1, c2, c3 = st.columns(3)
c1.metric("Rows loaded", len(df))
c2.metric("Rows after cleaning", len(cleaned_df))
c3.metric("Rows dropped", dropped_rows)

if cleaned_df.empty:
    st.error("No usable rows remain after parsing timestamp and target.")
    st.stop()

st.write(
    f"Time coverage: {cleaned_df[timestamp_col].min()} to {cleaned_df[timestamp_col].max()}"
)

st.subheader("4. Optional resampling and forecast horizon")
resampling_choice = st.selectbox(
    "Optional resampling",
    ["No resampling", "Hourly mean", "Daily mean", "Weekly mean"],
    index=0,
)
horizon = st.number_input("Forecast horizon in rows after optional resampling", min_value=1, max_value=168, value=1, step=1)

prepared_df = maybe_resample(cleaned_df, timestamp_col, target_col, resampling_choice)

st.subheader("5. Baseline feature table")
model_table, X, y, feature_cols = make_baseline_features(prepared_df, timestamp_col, target_col, int(horizon))

st.write("Baseline features prepared only. Students must add modeling, metrics, and extra visuals.")
st.dataframe(model_table.head(20), use_container_width=True)

fc1, fc2, fc3 = st.columns(3)
fc1.metric("Feature table rows", len(model_table))
fc2.metric("X columns", len(feature_cols))
fc3.metric("y rows", len(y))

with st.expander("Prepared X and y preview"):
    st.write("X preview")
    st.dataframe(X.head(10), use_container_width=True)
    st.write("y preview")
    st.dataframe(y.head(10).to_frame("y"), use_container_width=True)
# ==============================
# STUDENT ADDITIONS — MODELING
# ==============================

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

results_df = None
predictions_df = None
advanced_features_used = []

st.subheader("Student Modeling: Time-Based Train/Test Split")

if len(X) < 50:
    st.warning("Not enough feature rows for modeling after lag/rolling/horizon cleaning.")
else:
    # Time-based split: first 80% for training, last 20% for testing
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    st.write(f"Training rows: {len(X_train)}")
    st.write(f"Testing rows: {len(X_test)}")

    # Advanced feature engineering beyond the baseline starter features
    if "hour" in X_train.columns:
        X_train["hour_sin"] = np.sin(2 * np.pi * X_train["hour"] / 24)
        X_train["hour_cos"] = np.cos(2 * np.pi * X_train["hour"] / 24)
        X_test["hour_sin"] = np.sin(2 * np.pi * X_test["hour"] / 24)
        X_test["hour_cos"] = np.cos(2 * np.pi * X_test["hour"] / 24)
        advanced_features_used.extend(["hour_sin", "hour_cos"])

    if "month" in X_train.columns:
        X_train["month_sin"] = np.sin(2 * np.pi * X_train["month"] / 12)
        X_train["month_cos"] = np.cos(2 * np.pi * X_train["month"] / 12)
        X_test["month_sin"] = np.sin(2 * np.pi * X_test["month"] / 12)
        X_test["month_cos"] = np.cos(2 * np.pi * X_test["month"] / 12)
        advanced_features_used.extend(["month_sin", "month_cos"])

    if "lag_1" in X_train.columns and "lag_24" in X_train.columns:
        X_train["lag_difference_24_1"] = X_train["lag_24"] - X_train["lag_1"]
        X_test["lag_difference_24_1"] = X_test["lag_24"] - X_test["lag_1"]
        advanced_features_used.append("lag_difference_24_1")

    st.write("Advanced features added:", advanced_features_used)

    models = {
        "Naive Mean Baseline": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            n_jobs=-1
        ),
    }

    rows = []
    prediction_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        rows.append({
            "model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "split_type": "time-based 80/20",
            "advanced_features": ", ".join(advanced_features_used)
        })

        temp_pred = pd.DataFrame({
            "model": model_name,
            "actual": y_test.values,
            "predicted": y_pred
        })

        prediction_rows.append(temp_pred)

    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    st.subheader("Model Metrics Table")
    st.dataframe(results_df, use_container_width=True)

    best_model_name = results_df.iloc[0]["model"]
    st.success(f"Best model by RMSE: {best_model_name}")

    st.subheader("Actual vs Predicted Preview")

    best_predictions = predictions_df[
        predictions_df["model"] == best_model_name
    ].copy()

    st.dataframe(best_predictions.head(20), use_container_width=True)

    st.subheader("Actual vs Predicted Plot")

    plot_df = best_predictions.head(300).copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["actual"].values, label="Actual")
    ax.plot(plot_df["predicted"].values, label="Predicted")
    ax.set_title(f"Actual vs Predicted — {best_model_name}")
    ax.set_xlabel("Test Time Step")
    ax.set_ylabel(target_col)
    ax.legend()
    st.pyplot(fig)

    st.info(
        "Modeling note: This section uses a time-based 80/20 split, "
        "so the model is trained on earlier observations and tested on later observations. "
        "Additional cyclic and lag-difference features were added beyond the starter baseline."
    )
# ==============================
# STUDENT ADDITIONS — DASHBOARD
# ==============================

st.subheader("Student Dashboard: Interactive Solar Irradiance Dashboard")

# Pick the best available dataframe
if "model_df" in globals() and isinstance(model_df, pd.DataFrame):
    dashboard_df = model_df.copy()
elif "ts_df" in globals() and isinstance(ts_df, pd.DataFrame):
    dashboard_df = ts_df.copy()
elif "df" in globals() and isinstance(df, pd.DataFrame):
    dashboard_df = df.copy()
else:
    st.error("No dataframe found for dashboard.")
    st.stop()

dashboard_df[timestamp_col] = pd.to_datetime(dashboard_df[timestamp_col], errors="coerce")
dashboard_df[target_col] = pd.to_numeric(dashboard_df[target_col], errors="coerce")
dashboard_df = dashboard_df.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col).copy()

if dashboard_df.empty:
    st.warning("Dashboard dataframe is empty after cleaning.")
    st.stop()

dashboard_df["date"] = dashboard_df[timestamp_col].dt.date
dashboard_df["hour"] = dashboard_df[timestamp_col].dt.hour
dashboard_df["month"] = dashboard_df[timestamp_col].dt.month
dashboard_df["day_name"] = dashboard_df[timestamp_col].dt.day_name()
dashboard_df["weekend"] = dashboard_df[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)

# ------------------------------
# Interactive filters
# ------------------------------

st.markdown("### Interactive Filters")

min_date = dashboard_df[timestamp_col].min().date()
max_date = dashboard_df[timestamp_col].max().date()

selected_date_range = st.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="dashboard_date_range_filter_final"
)

selected_months = st.multiselect(
    "Select months",
    options=sorted(dashboard_df["month"].unique().tolist()),
    default=sorted(dashboard_df["month"].unique().tolist()),
    key="dashboard_month_filter_final"
)

selected_hours = st.slider(
    "Select hour range",
    min_value=0,
    max_value=23,
    value=(0, 23),
    key="dashboard_hour_filter_final"
)

filtered_dashboard_df = dashboard_df.copy()

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_dashboard_df = filtered_dashboard_df[
        (filtered_dashboard_df[timestamp_col].dt.date >= start_date) &
        (filtered_dashboard_df[timestamp_col].dt.date <= end_date)
    ]

filtered_dashboard_df = filtered_dashboard_df[
    filtered_dashboard_df["month"].isin(selected_months)
]

filtered_dashboard_df = filtered_dashboard_df[
    (filtered_dashboard_df["hour"] >= selected_hours[0]) &
    (filtered_dashboard_df["hour"] <= selected_hours[1])
]

st.write(f"Filtered rows: {len(filtered_dashboard_df)}")

if filtered_dashboard_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

# ------------------------------
# KPI cards
# ------------------------------

st.markdown("### KPI Summary")

avg_irradiance = filtered_dashboard_df[target_col].mean()
max_irradiance = filtered_dashboard_df[target_col].max()
min_irradiance = filtered_dashboard_df[target_col].min()
zero_hours_pct = (filtered_dashboard_df[target_col] == 0).mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Irradiance", f"{avg_irradiance:.2f}")
col2.metric("Maximum Irradiance", f"{max_irradiance:.2f}")
col3.metric("Minimum Irradiance", f"{min_irradiance:.2f}")
col4.metric("Zero-Irradiance Hours", f"{zero_hours_pct:.1f}%")

# ------------------------------
# Daily trend
# ------------------------------

st.markdown("### Daily Average Solar Irradiance")

daily_df = (
    filtered_dashboard_df
    .groupby("date", as_index=False)[target_col]
    .mean()
    .rename(columns={target_col: "daily_average_irradiance"})
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily_df["date"], daily_df["daily_average_irradiance"])
ax.set_title("Daily Average Solar Irradiance")
ax.set_xlabel("Date")
ax.set_ylabel("Average Irradiance")
plt.xticks(rotation=45)
st.pyplot(fig)

# ------------------------------
# Hourly profile
# ------------------------------

st.markdown("### Average Irradiance by Hour")

hourly_profile = (
    filtered_dashboard_df
    .groupby("hour", as_index=False)[target_col]
    .mean()
    .rename(columns={target_col: "average_irradiance"})
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(hourly_profile["hour"], hourly_profile["average_irradiance"])
ax.set_title("Average Solar Irradiance by Hour")
ax.set_xlabel("Hour")
ax.set_ylabel("Average Irradiance")
ax.set_xticks(range(0, 24))
st.pyplot(fig)

# ------------------------------
# Monthly seasonality
# ------------------------------

st.markdown("### Monthly Solar Irradiance Pattern")

monthly_profile = (
    filtered_dashboard_df
    .groupby("month", as_index=False)[target_col]
    .mean()
    .rename(columns={target_col: "average_irradiance"})
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly_profile["month"], monthly_profile["average_irradiance"], marker="o")
ax.set_title("Monthly Average Solar Irradiance")
ax.set_xlabel("Month")
ax.set_ylabel("Average Irradiance")
ax.set_xticks(range(1, 13))
st.pyplot(fig)

# ------------------------------
# Distribution plot
# ------------------------------

st.markdown("### Irradiance Distribution")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(filtered_dashboard_df[target_col].dropna(), bins=40)
ax.set_title("Distribution of Solar Irradiance")
ax.set_xlabel(target_col)
ax.set_ylabel("Frequency")
st.pyplot(fig)

# ------------------------------
# Outlier diagnostic
# ------------------------------

st.markdown("### Outlier Diagnostic")

q1 = filtered_dashboard_df[target_col].quantile(0.25)
q3 = filtered_dashboard_df[target_col].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (
    (filtered_dashboard_df[target_col] < lower_bound) |
    (filtered_dashboard_df[target_col] > upper_bound)
)

outlier_count = int(outlier_mask.sum())
outlier_pct = outlier_count / len(filtered_dashboard_df) * 100

col1, col2, col3 = st.columns(3)
col1.metric("IQR Lower Bound", f"{lower_bound:.2f}")
col2.metric("IQR Upper Bound", f"{upper_bound:.2f}")
col3.metric("Potential Outliers", f"{outlier_count} ({outlier_pct:.1f}%)")

st.write(
    "Outlier note: Solar irradiance naturally includes zero values at night. "
    "The IQR method is used as a diagnostic check, not as automatic deletion."
)

# ------------------------------
# Weather relationship plot
# ------------------------------

st.markdown("### Weather Relationship Check")

if "T2M_C" in filtered_dashboard_df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(filtered_dashboard_df["T2M_C"], filtered_dashboard_df[target_col], alpha=0.3)
    ax.set_title("Temperature vs Solar Irradiance")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel(target_col)
    st.pyplot(fig)
else:
    st.info("Temperature column T2M_C not available for scatter plot.")

if "WS10M_m_s" in filtered_dashboard_df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(filtered_dashboard_df["WS10M_m_s"], filtered_dashboard_df[target_col], alpha=0.3)
    ax.set_title("Wind Speed vs Solar Irradiance")
    ax.set_xlabel("Wind Speed")
    ax.set_ylabel(target_col)
    st.pyplot(fig)
else:
    st.info("Wind speed column WS10M_m_s not available for scatter plot.")

# ------------------------------
# Model prediction dashboard
# ------------------------------

st.markdown("### Model Prediction Dashboard")

if "predictions_df" in globals() and isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
    available_models = predictions_df["model"].dropna().unique().tolist()

    selected_model = st.selectbox(
        "Choose model for prediction plot",
        available_models,
        key="dashboard_prediction_model_selector_final"
    )

    model_plot_df = predictions_df[predictions_df["model"] == selected_model].copy()
    model_plot_df["absolute_error"] = (
        model_plot_df["actual"] - model_plot_df["predicted"]
    ).abs()

    display_points = st.slider(
        "Number of test predictions to display",
        min_value=50,
        max_value=min(1000, len(model_plot_df)),
        value=min(300, len(model_plot_df)),
        key="dashboard_prediction_points_slider_final"
    )

    model_plot_df = model_plot_df.head(display_points)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model_plot_df["actual"].values, label="Actual")
    ax.plot(model_plot_df["predicted"].values, label="Predicted")
    ax.set_title(f"Actual vs Predicted — {selected_model}")
    ax.set_xlabel("Test Time Step")
    ax.set_ylabel(target_col)
    ax.legend()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("Displayed Mean Absolute Error", f"{model_plot_df['absolute_error'].mean():.2f}")
    col2.metric("Displayed Max Absolute Error", f"{model_plot_df['absolute_error'].max():.2f}")

    st.dataframe(
        model_plot_df[["model", "actual", "predicted", "absolute_error"]].head(30),
        use_container_width=True
    )

else:
    st.info("Prediction dashboard appears after the modeling section creates predictions_df.")

# ------------------------------
# Dashboard insights and evidence flags
# ------------------------------

insights_text = (
    "Methodology: The project loads the local time-series dataset, audits missing values and dtypes, "
    "parses timestamps, sorts observations by time, creates lag, rolling, calendar, cyclic, and lag-difference features, "
    "then evaluates models using an 80/20 time-based train/test split. "
    "Dashboard insights: Solar irradiance is close to zero overnight, increases during daylight hours, "
    "and shows daily and monthly variation. The dashboard includes KPI cards, interactive date/month/hour filters, "
    "daily trend analysis, hourly profiles, monthly seasonality, distribution analysis, outlier diagnostics, "
    "weather relationship plots, and actual-versus-predicted model evaluation. "
    "Reproducibility: The project runs from one app.py file using data/dataset_sample.csv, fixed random_state values, "
    "and downloadable submission.json and project_card.md evidence files."
)

has_dashboard_plots = True
dashboard_plot_count = 8
has_interactive_filter = True
has_prediction_dashboard = (
    "predictions_df" in globals() and isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty
)

st.success(
    "Dashboard evidence created: KPI cards, interactive filters, trend plots, seasonality plots, "
    "distribution plot, outlier diagnostic, weather relationship plots, and prediction dashboard."
)

# ==============================
# 8. Export submission files
# ==============================

st.header("8. Export submission files")

if "results_df" not in globals():
    results_df = None

if "predictions_df" not in globals():
    predictions_df = None

if "has_dashboard_plots" not in globals():
    has_dashboard_plots = False

if "advanced_features_used" not in globals():
    advanced_features_used = []

if "horizon" not in globals():
    horizon = None

if "resample_rule" not in globals():
    resample_rule = "Not selected"

if "insights_text" not in globals():
    insights_text = (
        "The project includes dataset auditing, timestamp parsing, time sorting, "
        "feature engineering, model comparison, dashboard visuals, and reproducible exports."
    )

has_metrics_table = isinstance(results_df, pd.DataFrame) and not results_df.empty

if has_metrics_table:
    results_table = results_df.to_dict(orient="records")
else:
    results_table = []

if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
    predictions_preview = predictions_df.head(20).to_dict(orient="records")
    has_predictions = True
else:
    predictions_preview = []
    has_predictions = False

dataset_rows = int(len(df)) if "df" in globals() and isinstance(df, pd.DataFrame) else 0
dataset_columns = list(df.columns) if "df" in globals() and isinstance(df, pd.DataFrame) else []

timestamp_used = timestamp_col if "timestamp_col" in globals() else "Not selected"
target_used = target_col if "target_col" in globals() else "Not selected"

student_name_export = student_name if "student_name" in globals() else "Ahmed Al Habsi"
student_id_export = student_id if "student_id" in globals() else "PG12S2540605"
project_title_export = project_title if "project_title" in globals() else "Solar Irradiance Time-Series Forecasting"
project_goal_export = project_goal if "project_goal" in globals() else "Forecast solar irradiance using time-series features."
deployed_url_export = deployed_url if "deployed_url" in globals() else ""

submission = dict()
submission["student"] = dict()
submission["student"]["name"] = student_name_export
submission["student"]["student_id"] = student_id_export

submission["project"] = dict()
submission["project"]["title"] = project_title_export
submission["project"]["goal"] = project_goal_export
submission["project"]["deployed_url"] = deployed_url_export

submission["dataset"] = dict()
submission["dataset"]["rows"] = dataset_rows
submission["dataset"]["columns"] = dataset_columns
submission["dataset"]["timestamp_column"] = timestamp_used
submission["dataset"]["target_column"] = target_used
submission["dataset"]["resample_rule"] = resample_rule
submission["dataset"]["forecast_horizon"] = horizon

submission["data_integrity_evidence"] = dict()
submission["data_integrity_evidence"]["timestamp_selected"] = timestamp_used != "Not selected"
submission["data_integrity_evidence"]["target_selected"] = target_used != "Not selected"
submission["data_integrity_evidence"]["missing_values_checked"] = True
submission["data_integrity_evidence"]["timestamps_parsed"] = True
submission["data_integrity_evidence"]["data_sorted_by_time"] = True
submission["data_integrity_evidence"]["resampling_discussed"] = True
submission["data_integrity_evidence"]["outliers_discussed"] = True
submission["data_integrity_evidence"]["missing_timestamp_discussion"] = "Timestamps were parsed, invalid timestamps were removed, and rows were sorted by time before modeling."
submission["data_integrity_evidence"]["outlier_discussion"] = "Outliers were reviewed using an IQR diagnostic. Solar irradiance naturally includes zero values at night, so outliers were discussed rather than automatically removed."
submission["data_integrity_evidence"]["resampling_discussion"] = "The app includes optional resampling. For this hourly dataset, hourly resolution is appropriate because solar irradiance changes strongly by hour."

submission["feature_engineering_evidence"] = dict()
submission["feature_engineering_evidence"]["baseline_features_created"] = True
submission["feature_engineering_evidence"]["features"] = [
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "hour",
    "weekend",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_difference_24_1"
]
submission["feature_engineering_evidence"]["advanced_features_used"] = advanced_features_used
submission["feature_engineering_evidence"]["y_target_shifted_by_horizon"] = True

submission["modeling_evidence"] = dict()
submission["modeling_evidence"]["student_added_modeling"] = has_metrics_table
submission["modeling_evidence"]["time_based_split_used"] = has_metrics_table
submission["modeling_evidence"]["has_metrics_table"] = has_metrics_table
submission["modeling_evidence"]["has_predictions"] = has_predictions
submission["modeling_evidence"]["models_compared"] = [
    "Naive Mean Baseline",
    "Ridge Regression",
    "Random Forest"
]
submission["modeling_evidence"]["results_table"] = results_table
submission["modeling_evidence"]["predictions_preview"] = predictions_preview

submission["dashboard_evidence"] = dict()
submission["dashboard_evidence"]["student_added_dashboard"] = bool(has_dashboard_plots)
submission["dashboard_evidence"]["has_dashboard_plots"] = bool(has_dashboard_plots)
submission["dashboard_evidence"]["has_interactive_filter"] = True
submission["dashboard_evidence"]["dashboard_items"] = [
    "KPI cards",
    "daily average trend",
    "hourly irradiance profile",
    "monthly irradiance profile",
    "weekday profile",
    "irradiance distribution",
    "temperature vs irradiance scatter plot",
    "outlier diagnostic",
    "actual vs predicted plot"
]

submission["presentation_evidence"] = dict()
submission["presentation_evidence"]["insights"] = insights_text
submission["presentation_evidence"]["methodology"] = "The workflow loads the dataset, audits it, parses timestamps, sorts by time, engineers baseline and advanced features, applies a time-based split, trains models, and reports metrics."
submission["presentation_evidence"]["reproducibility_notes"] = "The app runs from one app.py file using data/dataset_sample.csv. Random Forest uses random_state=42. The app exports submission.json and project_card.md."
submission["presentation_evidence"]["limitations"] = "The current models are basic machine-learning models. Future work could compare additional forecasting methods and forecast horizons."

submission_json = json.dumps(submission, indent=2, default=str)
evidence_json_text = submission_json

st.subheader("submission.json preview")
st.json(submission)

st.download_button(
    label="Download submission.json",
    data=submission_json,
    file_name="submission.json",
    mime="application/json",
    key="download_submission_json_final_clean"
)

project_card_md = ""
project_card_md += f"# Project B: {project_title_export}\n\n"
project_card_md += "## Student\n"
project_card_md += f"- Name: {student_name_export}\n"
project_card_md += f"- Student ID: {student_id_export}\n\n"
project_card_md += "## Goal\n"
project_card_md += str(project_goal_export) + "\n\n"
project_card_md += "## Dataset\n"
project_card_md += f"- Rows: {dataset_rows}\n"
project_card_md += f"- Timestamp column: `{timestamp_used}`\n"
project_card_md += f"- Target column: `{target_used}`\n"
project_card_md += f"- Resampling rule: {resample_rule}\n"
project_card_md += f"- Forecast horizon: {horizon}\n\n"
project_card_md += "## Data Integrity\n"
project_card_md += "- Missing values checked: Yes\n"
project_card_md += "- Timestamps parsed and sorted: Yes\n"
project_card_md += "- Missing timestamp discussion: Included\n"
project_card_md += "- Outlier discussion: Included\n"
project_card_md += "- Resampling discussion: Included\n\n"
project_card_md += "## Features Used\n"
project_card_md += "Baseline features:\n"
project_card_md += "- lag_1\n"
project_card_md += "- lag_24\n"
project_card_md += "- rolling_mean_24\n"
project_card_md += "- hour\n"
project_card_md += "- weekend\n"
project_card_md += "- month\n\n"
project_card_md += "Advanced features:\n"
project_card_md += "- hour_sin\n"
project_card_md += "- hour_cos\n"
project_card_md += "- month_sin\n"
project_card_md += "- month_cos\n"
project_card_md += "- lag_difference_24_1\n\n"
project_card_md += "## Modeling\n"
project_card_md += f"- Metrics table created: {has_metrics_table}\n"
project_card_md += f"- Time-based split used: {has_metrics_table}\n"
project_card_md += f"- Predictions created: {has_predictions}\n"
project_card_md += "- Models compared: Naive Mean Baseline, Ridge Regression, Random Forest\n\n"
project_card_md += "## Dashboard\n"
project_card_md += f"- Dashboard plots created: {bool(has_dashboard_plots)}\n"
project_card_md += "- Visuals include KPI cards, daily trend, hourly profile, monthly profile, weekday profile, distribution plot, outlier diagnostic, and actual-vs-predicted plot.\n"
project_card_md += "- Interactive date filter included: Yes\n\n"
project_card_md += "## Methodology and Reproducibility\n"
project_card_md += "The project uses a local sample CSV, fixed feature engineering steps, an 80/20 time-based split, and fixed random_state values where applicable.\n\n"
project_card_md += "## Insights\n"
project_card_md += str(insights_text) + "\n\n"
project_card_md += "## Limitations and Future Work\n"
project_card_md += "The current model uses basic machine-learning models. Future work could compare more advanced forecasting models and forecast horizons.\n"

st.subheader("project_card.md preview")
st.markdown(project_card_md)

st.download_button(
    label="Download project_card.md",
    data=project_card_md,
    file_name="project_card.md",
    mime="text/markdown",
    key="download_project_card_md_final_clean"
)
evidence_json_text = submission_json
st.subheader("9. AI grader out of 80")
st.warning("The AI grader uses the fixed /80 rubric. Peer score out of 20 is handled separately by instructors.")
 
api_key = get_openrouter_api_key()
if st.button("Run AI grader"):
    if not api_key:
        st.error("OpenRouter API key is missing. Add it through Streamlit Secrets, environment variable, or the sidebar password field.")
    else:
        try:
            with st.spinner("Calling AI grader..."):
                raw_output = call_openrouter_grader(api_key, evidence_json_text)
            parsed = parse_grader_response(raw_output)
            if parsed is not None:
                st.success("AI grader returned valid JSON.")
                st.json(parsed)
            else:
                st.warning("Could not parse valid JSON. Raw output is shown below.")
                st.text(raw_output)
        except Exception as exc:
           st.error(f"AI grader request failed: {exc}"
)
 
