# ================================
# GDP FORECASTING STREAMLIT APP
# ================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="🌍 GDP AI Forecast Engine",
    layout="wide"
)

# ================================
# GLOBALS
# ================================
DATA_FILE = "data.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ================================
# DATA LOADING (SAFE)
# ================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error("❌ data.csv not found")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    required_cols = {"Country", "Year", "Activity", "Value"}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ CSV must contain {required_cols}")
        st.stop()

    df = df.dropna(subset=["Country", "Year", "Activity", "Value"])
    df["Year"] = df["Year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    return df

# ================================
# PREPROCESSING
# ================================
def preprocess(df):
    # Remove extreme outliers (IQR)
    q1, q3 = df["Value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df["Value"] >= q1 - 1.5 * iqr) &
            (df["Value"] <= q3 + 1.5 * iqr)]

    df = df.sort_values(["Country", "Activity", "Year"])

    # Lag feature
    df["Lag_1"] = df.groupby(["Country", "Activity"])["Value"].shift(1)
    df = df.dropna(subset=["Lag_1"])

    return df

# ================================
# MODEL TRAINING
# ================================
@st.cache_resource
def train_model(df):
    le_country = LabelEncoder()
    le_activity = LabelEncoder()

    df["Country_Enc"] = le_country.fit_transform(df["Country"])
    df["Activity_Enc"] = le_activity.fit_transform(df["Activity"])

    X = df[["Year", "Country_Enc", "Activity_Enc", "Lag_1"]]

    # LOG SCALE TARGET (CRITICAL)
    y = np.log1p(df["Value"])

    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1
    )

    model.fit(X, y)

    return model, le_country, le_activity, df

# ================================streamlit run app_forecast_2035.py
# LOAD EVERYTHING
# ================================
raw_df = load_data()
clean_df = preprocess(raw_df)
model, le_country, le_activity, df_train = train_model(clean_df)

# ================================
# UI
# ================================
st.title("🌍 Global GDP Forecasting Dashboard")
st.markdown("Professional, noise-free AI predictions")

# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.header("🔧 Forecast Controls")

country = st.sidebar.selectbox(
    "🌏 Country",
    sorted(le_country.classes_)
)

activity = st.sidebar.selectbox(
    "🛠️ Activity Code",
    sorted(le_activity.classes_)
)

target_year = st.sidebar.slider(
    "📅 Forecast Until Year",
    int(df_train["Year"].max()) + 1,
    2035,
    2030
)

# ================================
# VALIDATION
# ================================
hist = df_train[
    (df_train["Country"] == country) &
    (df_train["Activity"] == activity)
].sort_values("Year")

if hist.empty:
    st.error("❌ Not enough historical data for this country & activity.")
    st.stop()

# ================================
# FORECAST
# ================================
last_year = int(hist["Year"].iloc[-1])
last_value = hist["Value"].iloc[-1]
c_enc = le_country.transform([country])[0]
a_enc = le_activity.transform([activity])[0]

years = []
preds = []

temp_val = last_value

for yr in range(last_year + 1, target_year + 1):
    log_pred = model.predict([[yr, c_enc, a_enc, temp_val]])[0]
    pred = np.expm1(log_pred)
    years.append(yr)
    preds.append(pred)
    temp_val = pred

forecast_df = pd.DataFrame({
    "Year": years,
    "Value": preds
})

combined = pd.concat([
    hist[["Year", "Value"]],
    forecast_df
])

# ================================
# MAIN DASHBOARD
# ================================
col1, col2 = st.columns([3, 1])

with col1:
    fig = px.line(
        combined,
        x="Year",
        y="Value",
        title=f"{country} — Activity {activity} GDP Forecast",
        markers=True
    )

    fig.add_vrect(
        x0=last_year,
        x1=target_year,
        fillcolor="green",
        opacity=0.1,
        annotation_text="AI Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric(
        "📈 Final Predicted Value",
        f"${preds[-1]:,.2f}"
    )

    growth = ((preds[-1] - last_value) / last_value) * 100
    st.metric("📊 Growth %", f"{growth:.2f}%")

# ================================
# PROFESSIONAL TOP-10 CHARTS
# ================================
st.markdown("---")
st.subheader("📊 Clean Top-10 Analytics")

top_country = (
    df_train.groupby("Country")["Value"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=top_country.values,
    y=top_country.index,
    ax=ax
)
ax.set_title("Top 10 Countries by Mean GDP Value")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/top10_countries.png")
st.pyplot(fig)

# ================================
# DONE
# ================================
st.success("✅ Forecast generated successfully")
