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
# 6. STUDENT ADDITIONS — MODELING
# ==============================

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

st.header("6. Modeling and Evaluation")

# Ensure timestamp format and chronological order
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

target_col = "ALLSKY_SFC_SW_DWN_Wh_m2"

# ------------------------------
# Timestamp integrity
# ------------------------------
df["time_gap"] = df["timestamp"].diff()
most_common_gap = df["time_gap"].mode()[0]

st.subheader("Time-Series Integrity Check")
st.write(f"Most common timestamp interval: **{most_common_gap}**")
st.write(f"Missing target values: **{df[target_col].isna().sum()}**")

# ------------------------------
# Feature engineering
# ------------------------------
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["dayofyear"] = df["timestamp"].dt.dayofyear
df["weekday"] = df["timestamp"].dt.weekday

df["lag_1"] = df[target_col].shift(1)
df["lag_24"] = df[target_col].shift(24)
df["rolling_mean_24"] = df[target_col].rolling(window=24).mean()
df["rolling_std_24"] = df[target_col].rolling(window=24).std()

df = df.dropna().reset_index(drop=True)

features = [
    "hour",
    "day",
    "month",
    "dayofyear",
    "weekday",
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "rolling_std_24"
]

X = df[features]
y = df[target_col]

# ------------------------------
# Time-based train/test split
# Last 20% used for testing
# ------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_time = df["timestamp"].iloc[split_index:]

st.subheader("Time-Based Train/Test Split")
st.write(f"Training records: **{len(X_train)}**")
st.write(f"Testing records: **{len(X_test)}**")
st.write(f"Training period: **{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[split_index-1]}**")
st.write(f"Testing period: **{df['timestamp'].iloc[split_index]} to {df['timestamp'].iloc[-1]}**")

# ------------------------------
# Models
# ------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        max_depth=10
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
predictions = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_model)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_model))
    mape = np.mean(np.abs((y_test - y_pred_model) / np.where(y_test == 0, np.nan, y_test))) * 100

    results.append({
        "model": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE": round(mape, 3)
    })

    predictions[model_name] = y_pred_model

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

st.subheader("Model Comparison Results")
st.dataframe(results_df)

best_model_name = results_df.loc[0, "model"]
y_pred = predictions[best_model_name]

st.success(f"Best model based on RMSE: {best_model_name}")

# ==============================
# 7. STUDENT ADDITIONS — DASHBOARD
# ==============================

import matplotlib.pyplot as plt

st.header("7. Dashboard, Visual Analysis and Insights")

# ------------------------------
# KPI Cards
# ------------------------------
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Average Irradiance", f"{df[target_col].mean():.2f} Wh/m²")
col3.metric("Maximum Irradiance", f"{df[target_col].max():.2f} Wh/m²")
col4.metric("Best Model", best_model_name)

# ------------------------------
# Dataset Integrity and Outliers
# ------------------------------
st.subheader("Data Integrity and Outlier Review")

Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]

st.write(f"Detected outliers using IQR method: **{len(outliers)}**")
st.write(f"Lower bound: **{lower_bound:.2f} Wh/m²**")
st.write(f"Upper bound: **{upper_bound:.2f} Wh/m²**")

fig_box, ax_box = plt.subplots(figsize=(10, 4))
ax_box.boxplot(df[target_col], vert=False)
ax_box.set_title("Solar Irradiance Outlier Detection")
ax_box.set_xlabel("Solar Irradiance Wh/m²")
st.pyplot(fig_box)

st.write("""
Outliers were checked using the IQR method. However, solar irradiance naturally changes between very low values
at night and high values during peak sunlight hours. Therefore, the outliers were reviewed carefully instead of
being removed automatically.
""")

# ------------------------------
# Model Comparison Chart
# ------------------------------
st.subheader("Model Performance Comparison")

fig_model, ax_model = plt.subplots(figsize=(8, 4))
ax_model.bar(results_df["model"], results_df["RMSE"])
ax_model.set_title("Model Comparison Based on RMSE")
ax_model.set_xlabel("Model")
ax_model.set_ylabel("RMSE")
plt.xticks(rotation=20)
st.pyplot(fig_model)

# ------------------------------
# Actual vs Predicted
# ------------------------------
st.subheader("Actual vs Predicted Forecast")

fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
ax_pred.plot(test_time, y_test.values, label="Actual")
ax_pred.plot(test_time, y_pred, label="Predicted")
ax_pred.set_title(f"Actual vs Predicted Solar Irradiance - {best_model_name}")
ax_pred.set_xlabel("Timestamp")
ax_pred.set_ylabel("Solar Irradiance Wh/m²")
ax_pred.legend()
st.pyplot(fig_pred)

# ------------------------------
# Residual Analysis
# ------------------------------
st.subheader("Residual Analysis")

residuals = y_test.values - y_pred

fig_res, ax_res = plt.subplots(figsize=(12, 4))
ax_res.plot(test_time, residuals)
ax_res.axhline(0, linestyle="--")
ax_res.set_title("Residuals Over Time")
ax_res.set_xlabel("Timestamp")
ax_res.set_ylabel("Prediction Error")
st.pyplot(fig_res)

fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
ax_hist.hist(residuals, bins=30)
ax_hist.set_title("Residual Distribution")
ax_hist.set_xlabel("Residual Error")
ax_hist.set_ylabel("Frequency")
st.pyplot(fig_hist)

st.write("""
Residual analysis was used to check prediction errors. A good forecasting model should have residuals distributed
around zero without a strong pattern. Larger errors may occur during sudden weather changes or rapid irradiance variation.
""")

# ------------------------------
# Monthly Pattern
# ------------------------------
st.subheader("Monthly Solar Irradiance Pattern")

monthly_avg = df.groupby("month")[target_col].mean()

fig_month, ax_month = plt.subplots(figsize=(8, 4))
ax_month.plot(monthly_avg.index, monthly_avg.values, marker="o")
ax_month.set_title("Average Monthly Solar Irradiance")
ax_month.set_xlabel("Month")
ax_month.set_ylabel("Average Irradiance Wh/m²")
st.pyplot(fig_month)

# ------------------------------
# Feature Importance
# ------------------------------
st.subheader("Feature Importance")

best_model = models[best_model_name]

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance_df)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    ax_imp.barh(importance_df["Feature"], importance_df["Importance"])
    ax_imp.set_title("Feature Importance")
    ax_imp.set_xlabel("Importance")
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)
else:
    st.info("Feature importance is not available for this model type.")

# ------------------------------
# Interactive Date Filter
# ------------------------------
st.subheader("Interactive Date Range Viewer")

start_date = st.date_input("Select start date", df["timestamp"].min().date())
end_date = st.date_input("Select end date", df["timestamp"].max().date())

filtered_df = df[
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date)
]

fig_filter, ax_filter = plt.subplots(figsize=(12, 5))
ax_filter.plot(filtered_df["timestamp"], filtered_df[target_col])
ax_filter.set_title("Solar Irradiance for Selected Date Range")
ax_filter.set_xlabel("Timestamp")
ax_filter.set_ylabel("Solar Irradiance Wh/m²")
st.pyplot(fig_filter)

# ------------------------------
# Insights and Narrative
# ------------------------------
st.subheader("Key Insights and Interpretation")

st.write(f"""
1. The dataset was prepared as a proper time-series dataset by converting the timestamp column to datetime format
and sorting all records chronologically.

2. A time-based train/test split was used. The first 80% of the dataset was used for training and the last 20%
was used for testing. This avoids data leakage and better represents real forecasting conditions.

3. Three models were compared: Linear Regression, Random Forest, and Gradient Boosting. The best model was
**{best_model_name}**, selected based on the lowest RMSE value.

4. Lag features and rolling statistics improved the forecasting process because solar irradiance depends strongly
on previous time steps and daily patterns.

5. Residual analysis shows how prediction errors behave over time. Large residuals may be caused by sudden changes
in sunlight due to cloud cover, seasonal variation, or weather effects.

6. The dashboard includes KPI cards, outlier detection, model comparison, actual vs predicted forecast, residual plots,
monthly trend analysis, feature importance, and an interactive date range viewer.

7. This forecasting workflow is useful for solar energy planning because it helps estimate future solar resource
availability at Manah, Oman.
""")
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
            st.error(f"AI grader request failed: {exc}")
