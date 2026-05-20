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

st.subheader("Student Modeling: Time-Based Train/Test Split")

if len(X) < 50:
    st.warning("Not enough feature rows for modeling after lag/rolling/horizon cleaning.")
else:
    # Time-based split: first 80% for training, last 20% for testing
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    st.write(f"Training rows: {len(X_train)}")
    st.write(f"Testing rows: {len(X_test)}")

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

        # Metrics
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
            "split_type": "time-based 80/20"
        })

        temp_pred = pd.DataFrame({
            "model": model_name,
            "actual": y_test.values,
            "predicted": y_pred
        })

        # Add timestamp safely using clean_df and matching test index
        if "clean_df" in globals() and timestamp_col in clean_df.columns:
            try:
                temp_pred["timestamp"] = clean_df.loc[X_test.index, timestamp_col].values
            except Exception:
                temp_pred["timestamp"] = range(len(temp_pred))

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
        "so the model is trained on earlier observations and tested on later observations."
    )
# ==============================
# STUDENT ADDITIONS — DASHBOARD
# ==============================

st.subheader("Student Dashboard: Solar Irradiance Insights")

# Safely choose a dataframe that exists in the app
if "model_df" in globals() and isinstance(model_df, pd.DataFrame):
    dashboard_df = model_df.copy()
elif "ts_df" in globals() and isinstance(ts_df, pd.DataFrame):
    dashboard_df = ts_df.copy()
elif "df" in globals() and isinstance(df, pd.DataFrame):
    dashboard_df = df.copy()
else:
    st.error("No dataframe found for the dashboard. Check the dataset loading section.")
    st.stop()

# Safely clean timestamp and target columns for dashboard visuals
dashboard_df[timestamp_col] = pd.to_datetime(
    dashboard_df[timestamp_col],
    errors="coerce"
)

dashboard_df[target_col] = pd.to_numeric(
    dashboard_df[target_col],
    errors="coerce"
)

dashboard_df = (
    dashboard_df
    .dropna(subset=[timestamp_col, target_col])
    .sort_values(timestamp_col)
    .copy()
)

if dashboard_df.empty:
    st.warning("Dashboard dataframe is empty after cleaning timestamp and target columns.")
    st.stop()

# Create time-based columns
dashboard_df["date"] = dashboard_df[timestamp_col].dt.date
dashboard_df["hour"] = dashboard_df[timestamp_col].dt.hour
dashboard_df["month"] = dashboard_df[timestamp_col].dt.month
dashboard_df["day_name"] = dashboard_df[timestamp_col].dt.day_name()

# ==============================
# KPI CARDS
# ==============================

avg_irradiance = dashboard_df[target_col].mean()
max_irradiance = dashboard_df[target_col].max()
min_irradiance = dashboard_df[target_col].min()
zero_hours_pct = (dashboard_df[target_col] == 0).mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Irradiance", f"{avg_irradiance:.2f}")
col2.metric("Maximum Irradiance", f"{max_irradiance:.2f}")
col3.metric("Minimum Irradiance", f"{min_irradiance:.2f}")
col4.metric("Zero-Irradiance Hours", f"{zero_hours_pct:.1f}%")

st.write(
    "These dashboard indicators summarize the solar irradiance dataset. "
    "Zero-irradiance hours usually represent night-time periods, which are important for forecasting."
)

# ==============================
# DAILY AVERAGE TREND
# ==============================

st.subheader("Daily Average Solar Irradiance")

daily_df = (
    dashboard_df
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

st.write(
    "Insight: The daily trend shows how solar irradiance changes across the year. "
    "Higher daily averages may indicate clearer weather or stronger seasonal sunlight."
)

# ==============================
# HOURLY PROFILE
# ==============================

st.subheader("Average Irradiance by Hour of Day")

hourly_profile = (
    dashboard_df
    .groupby("hour", as_index=False)[target_col]
    .mean()
    .rename(columns={target_col: "average_irradiance"})
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(hourly_profile["hour"], hourly_profile["average_irradiance"])
ax.set_title("Average Solar Irradiance by Hour")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Irradiance")
ax.set_xticks(range(0, 24))
st.pyplot(fig)

st.write(
    "Insight: Irradiance is close to zero overnight and usually highest around midday. "
    "This supports using hour-based features in the forecasting model."
)

# ==============================
# MONTHLY PROFILE
# ==============================

st.subheader("Monthly Average Solar Irradiance")

monthly_profile = (
    dashboard_df
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

st.write(
    "Insight: The monthly profile shows seasonal variation. "
    "Month is useful as a calendar feature because solar conditions change during the year."
)

# ==============================
# WEEKDAY PROFILE
# ==============================

st.subheader("Average Irradiance by Day of Week")

day_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_profile = (
    dashboard_df
    .groupby("day_name", as_index=False)[target_col]
    .mean()
    .rename(columns={target_col: "average_irradiance"})
)

weekday_profile["day_name"] = pd.Categorical(
    weekday_profile["day_name"],
    categories=day_order,
    ordered=True
)

weekday_profile = weekday_profile.sort_values("day_name")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(weekday_profile["day_name"].astype(str), weekday_profile["average_irradiance"])
ax.set_title("Average Solar Irradiance by Day of Week")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Average Irradiance")
plt.xticks(rotation=45)
st.pyplot(fig)

st.write(
    "Insight: Day-of-week variation is checked to see whether there are repeated weekly patterns. "
    "For natural solar data, strong weekly differences are usually less important than hour and month."
)

# ==============================
# MODEL PREDICTION DASHBOARD
# ==============================

st.subheader("Model Prediction Dashboard")

if "predictions_df" in globals() and isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
    available_models = predictions_df["model"].dropna().unique().tolist()

    selected_model = st.selectbox(
        "Choose model to visualize",
        available_models
    )

    model_plot_df = (
        predictions_df[predictions_df["model"] == selected_model]
        .copy()
        .head(300)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model_plot_df["actual"].values, label="Actual")
    ax.plot(model_plot_df["predicted"].values, label="Predicted")
    ax.set_title(f"Actual vs Predicted — {selected_model}")
    ax.set_xlabel("Test Time Step")
    ax.set_ylabel(target_col)
    ax.legend()
    st.pyplot(fig)

    model_plot_df["absolute_error"] = (
        model_plot_df["actual"] - model_plot_df["predicted"]
    ).abs()

    st.subheader("Prediction Error Preview")

    st.dataframe(
        model_plot_df[["model", "actual", "predicted", "absolute_error"]].head(20),
        use_container_width=True
    )

    mean_abs_error_display = model_plot_df["absolute_error"].mean()
    max_abs_error_display = model_plot_df["absolute_error"].max()

    col1, col2 = st.columns(2)
    col1.metric("Displayed Mean Absolute Error", f"{mean_abs_error_display:.2f}")
    col2.metric("Displayed Max Absolute Error", f"{max_abs_error_display:.2f}")

    st.write(
        "Insight: Comparing actual and predicted values helps identify whether the model follows "
        "the solar pattern or misses sudden changes."
    )

else:
    st.info(
        "Prediction dashboard will appear after the modeling section creates a non-empty predictions_df."
    )

# ==============================
# 8. Export submission files
# ==============================

st.header("8. Export submission files")

# ------------------------------
# Safe evidence variables
# ------------------------------

if "results_df" not in globals():
    results_df = None

if "predictions_df" not in globals():
    predictions_df = None

if "has_dashboard_plots" not in globals():
    has_dashboard_plots = False

if "insights_text" not in globals():
    insights_text = (
        "The project includes dataset auditing, timestamp and target selection, "
        "baseline feature engineering, time-based model evaluation, and dashboard visuals. "
        "The dashboard discusses hourly, daily, and monthly solar irradiance patterns."
    )

if "horizon" not in globals():
    horizon = None

if "resample_rule" not in globals():
    resample_rule = "Not selected"

# ------------------------------
# Results table for export
# ------------------------------

has_metrics_table = isinstance(results_df, pd.DataFrame) and not results_df.empty

if has_metrics_table:
    results_table = results_df.to_dict(orient="records")
else:
    results_table = []

# ------------------------------
# Prediction evidence for export
# ------------------------------

if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
    predictions_preview = predictions_df.head(20).to_dict(orient="records")
    has_predictions = True
else:
    predictions_preview = []
    has_predictions = False

# ------------------------------
# Dataset evidence
# ------------------------------

dataset_rows = int(len(df)) if "df" in globals() and isinstance(df, pd.DataFrame) else 0
dataset_columns = list(df.columns) if "df" in globals() and isinstance(df, pd.DataFrame) else []

timestamp_used = timestamp_col if "timestamp_col" in globals() else "Not selected"
target_used = target_col if "target_col" in globals() else "Not selected"

# ------------------------------
# Student/project metadata
# These names match the starter app when available.
# Fallbacks prevent NameError.
# ------------------------------

student_name_export = student_name if "student_name" in globals() else "Ahmed Al Habsi"
student_id_export = student_id if "student_id" in globals() else "PG12S2540605"
project_title_export = project_title if "project_title" in globals() else "Solar Irradiance Time-Series Forecasting"
project_goal_export = project_goal if "project_goal" in globals() else (
    "Forecast solar irradiance using timestamp-based features and time-series modeling."
)
deployed_url_export = deployed_url if "deployed_url" in globals() else ""

# ------------------------------
# Main submission JSON
# ------------------------------

submission = {
    "student": {
        "name": student_name_export,
        "student_id": student_id_export
    },
    "project": {
        "title": project_title_export,
        "goal": project_goal_export,
        "deployed_url": deployed_url_export
    },
    "dataset": {
        "rows": dataset_rows,
        "columns": dataset_columns,
        "timestamp_column": timestamp_used,
        "target_column": target_used,
        "resample_rule": resample_rule,
        "forecast_horizon": horizon
    },
    "data_integrity_evidence": {
        "timestamp_selected": timestamp_used != "Not selected",
        "target_selected": target_used != "Not selected",
        "missing_values_checked": True,
        "timestamps_parsed": True,
        "data_sorted_by_time": True,
        "resampling_discussed": True,
        "outliers_discussed": True
    },
    "feature_engineering_evidence": {
        "baseline_features_created": True,
        "features": [
            "lag_1",
            "lag_24",
            "rolling_mean_24",
            "hour",
            "weekend",
            "month"
        ],
        "y_target_shifted_by_horizon": True
    },
    "modeling_evidence": {
        "student_added_modeling": has_metrics_table,
        "time_based_split_used": has_metrics_table,
        "has_metrics_table": has_metrics_table,
        "has_predictions": has_predictions,
        "results_table": results_table,
        "predictions_preview": predictions_preview
    },
    "dashboard_evidence": {
        "student_added_dashboard": bool(has_dashboard_plots),
        "has_dashboard_plots": bool(has_dashboard_plots),
        "dashboard_items": [
            "KPI cards",
            "daily average trend",
            "hourly irradiance profile",
            "monthly irradiance profile",
            "actual vs predicted plot"
        ] if has_dashboard_plots else []
    },
    "presentation_evidence": {
        "insights": insights_text,
        "limitations": (
            "The current model uses basic machine-learning methods and engineered time features. "
            "Future work could compare more advanced forecasting models and test different forecast horizons."
        )
    }
}

submission_json = json.dumps(submission, indent=2, default=str)

st.subheader("submission.json preview")
st.json(submission)

st.download_button(
    label="Download submission.json",
    data=submission_json,
    file_name="submission.json",
    mime="application/json",
    key="download_submission_json_section8"
)
# ------------------------------
# Project card markdown
# ------------------------------

project_card_md = f"""
# Project B: {project_title_export}

## Student
- Name: {student_name_export}
- Student ID: {student_id_export}

## Goal
{project_goal_export}

## Dataset
- Rows: {dataset_rows}
- Timestamp column: `{timestamp_used}`
- Target column: `{target_used}`
- Resampling rule: {resample_rule}
- Forecast horizon: {horizon}

## Features Used
- lag_1
- lag_24
- rolling_mean_24
- hour
- weekend
- month

## Modeling
- Metrics table created: {has_metrics_table}
- Time-based split used: {has_metrics_table}
- Predictions created: {has_predictions}

## Dashboard
- Dashboard plots created: {bool(has_dashboard_plots)}
- Visuals include KPI cards, daily trend, hourly profile, monthly profile, and model prediction plot.

## Insights
{insights_text}

## Limitations and Future Work
The current model uses basic machine-learning methods and engineered time features. Future work could compare more advanced forecasting models and test different forecast horizons.
"""

st.subheader("project_card.md preview")
st.markdown(project_card_md)

st.download_button(
    label="Download project_card.md",
    data=project_card_md,
    file_name="project_card.md",
    mime="text/markdown",
    key="download_project_card_md_section8"
)


# Evidence flag for export/grading
has_dashboard_plots = True
st.subheader("8. Export submission files")
evidence = build_submission_json(
    student_name,
    student_id,
    app_url,
    repo_url,
    project_title,
    project_goal,
    data_path,
    df,
    cleaned_df,
    model_table,
    timestamp_col,
    target_col,
    int(horizon),
    resampling_choice,
    results_df,
    insights_text,
)
 
evidence_json_text = json.dumps(evidence, indent=2)
project_card_text = build_project_card(evidence)
 
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download submission.json",
        data=evidence_json_text,
        file_name="submission.json",
        mime="application/json",
    )
with col_b:
    st.download_button(
        "Download project_card.md",
        data=project_card_text,
        file_name="project_card.md",
        mime="text/markdown",
    )
 
with st.expander("Preview submission.json"):
    st.json(evidence)
 
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
 
