# ================================
# GDP FORECASTING STREAMLIT APP (2025-2035)
# ================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder

# Google Gemini API
import google.generativeai as genai

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="🌍 GDP AI Forecast Engine 2035",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# GEMINI API SETUP
# ================================
GEMINI_API_KEY = "AIzaSyDoCOpDeJzZzQPCHokZLYzzO6CE-wxcUDk"
genai.configure(api_key=GEMINI_API_KEY)

# Custom CSS for professional styling
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
    }
    h1, h2, h3 {
        color: #003366;
    }
    .stSelectbox, .stSlider {
        color: #1a1a1a;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-left: 5px solid #0066cc;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================
# GLOBALS
# ================================
DATA_FILE = "data_cleaned.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ================================
# DATA LOADING
# ================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ {DATA_FILE} not found")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    # Identify required columns (flexible naming)
    required_cols = {"Country", "Year", "Activity", "Value"}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ CSV must contain columns: {required_cols}")
        st.stop()

    # Clean data
    df = df.dropna(subset=["Country", "Year", "Activity", "Value"])
    df["Year"] = df["Year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    return df

# ================================
# DATA PREPROCESSING
# ================================
def preprocess(df):
    """Remove outliers using IQR and create lag features"""
    
    # Remove extreme outliers (IQR method)
    q1, q3 = df["Value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df["Value"] >= q1 - 1.5 * iqr) &
            (df["Value"] <= q3 + 1.5 * iqr)]

    # Sort for lag creation
    df = df.sort_values(["Country", "Activity", "Year"])

    # Create Lag_1 feature (previous year's value)
    df["Lag_1"] = df.groupby(["Country", "Activity"])["Value"].shift(1)
    
    # For countries/activities with only 1 record, use the value itself as lag (bootstrap)
    df["Lag_1"] = df["Lag_1"].fillna(df["Value"])
    
    return df

# ================================
# MODEL TRAINING (CORRECTED)
# ================================
@st.cache_resource
def train_model(df):
    """Train ExtraTreesRegressor with safe log transform"""
    
    # Encode categorical variables
    le_country = LabelEncoder()
    le_activity = LabelEncoder()

    df["Country_Enc"] = le_country.fit_transform(df["Country"])
    df["Activity_Enc"] = le_activity.fit_transform(df["Activity"])

    X = df[["Year", "Country_Enc", "Activity_Enc", "Lag_1"]]

    # ---------- CRITICAL FIX ----------
    # Safe log transform (handles negative/zero values)
    min_val = df["Value"].min()
    shift = abs(min_val) + 1 if min_val <= 0 else 0

    df["Value_Shifted"] = df["Value"] + shift
    df = df[df["Value_Shifted"] > 0]  # final safety check

    y = np.log(df["Value_Shifted"])
    # ----------------------------------

    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1
    )

    model.fit(X.loc[df.index], y)

    return model, le_country, le_activity, df, shift

# ================================
# AI INSIGHT GENERATION (GEMINI)
# ================================
@st.cache_data
def generate_insights(country, activity, current_val, projected_val, growth_pct, forecast_df):
    """Generate AI-powered insights using Google Gemini"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare data context
        years_list = ", ".join([f"{int(yr)}" for yr in forecast_df["Year"].head(5).tolist()])
        values_list = ", ".join([f"${v:,.0f}" for v in forecast_df["Predicted_Value"].head(5).tolist()])
        
        prompt = f"""
        You are an expert economic analyst. Based on this industrial output forecast data, provide 5 KEY INSIGHTS:
        
        📊 FORECAST DATA:
        - Country: {country}
        - Activity Sector: {activity}
        - Current Value (2024): ${current_val:,.2f}
        - Projected Value (2035): ${projected_val:,.2f}
        - Growth Rate: {growth_pct:.2f}%
        - Sample Years: {years_list}
        - Sample Projections: {values_list}
        
        Please provide:
        1. 🎯 GROWTH TRAJECTORY: Explain the growth pattern and what it means economically
        2. 💼 SECTORAL IMPACT: How will this growth affect the {activity} sector in {country}?
        3. 📈 COMPETITIVE POSITION: Where does this place {country} in global {activity} markets?
        4. ⚠️ RISKS & CHALLENGES: What factors could impact this forecast?
        5. 🚀 OPPORTUNITIES: What strategic initiatives should be considered?
        
        Be concise (2-3 sentences per insight) and data-driven.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Unable to generate insights: {str(e)}"

# ================================
# POLICY RECOMMENDATIONS (GEMINI)
# ================================
@st.cache_data
def generate_recommendations(country, activity, growth_pct, projected_val):
    """Generate policy and strategy recommendations"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        As a development policy expert, provide 3 ACTIONABLE RECOMMENDATIONS for {country}'s {activity} sector:
        
        Context:
        - Projected Growth (2025-2035): {growth_pct:.2f}%
        - Target Industrial Output Value: ${projected_val:,.0f}
        
        Format your response as:
        1. 🎯 RECOMMENDATION: [Clear action]
           💡 BENEFIT: [Expected outcome]
           📋 IMPLEMENTATION: [How to execute]
        
        2. 🎯 RECOMMENDATION: [Clear action]
           💡 BENEFIT: [Expected outcome]
           📋 IMPLEMENTATION: [How to execute]
        
        3. 🎯 RECOMMENDATION: [Clear action]
           💡 BENEFIT: [Expected outcome]
           📋 IMPLEMENTATION: [How to execute]
        
        Be specific, measurable, and aligned with sustainable development.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Unable to generate recommendations: {str(e)}"

# ================================
# LOAD AND TRAIN
# ================================
raw_df = load_data()
clean_df = preprocess(raw_df)
model, le_country, le_activity, df_train, shift_value = train_model(clean_df)

# ================================
# UI TITLE
# ================================
st.title("🌍 Global GDP Forecasting Dashboard (2025-2035)")
st.markdown("Professional AI-driven predictions using recursive forecasting methodology")

# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.header("🔧 Forecast Configuration")

# Get available countries in training data
available_countries = sorted(df_train["Country"].unique())
country = st.sidebar.selectbox(
    "🌏 Select Country",
    available_countries,
    index=0
)

# Filter activities based on selected country
available_activities = sorted(df_train[df_train["Country"] == country]["Activity"].unique())
activity = st.sidebar.selectbox(
    "🛠️ Select Activity Code",
    available_activities,
    index=0
)

target_year = st.sidebar.slider(
    "📅 Forecast Until Year",
    min_value=int(df_train["Year"].max()) + 1,
    max_value=2035,
    value=2035
)

st.sidebar.markdown("---")
st.sidebar.info("💡 The recursive loop feeds each year's prediction back as the next year's input, capturing economic path-dependency.")

# ================================
# VALIDATION & HISTORICAL DATA
# ================================
hist = df_train[
    (df_train["Country"] == country) &
    (df_train["Activity"] == activity)
].sort_values("Year").copy()

if hist.empty:
    st.error(f"❌ No historical data available for {country} — Activity {activity}")
    st.stop()

# ================================
# RECURSIVE FORECASTING (2025-2035)
# ================================
last_year = int(hist["Year"].iloc[-1])
last_value = hist["Value"].iloc[-1]

c_enc = le_country.transform([country])[0]
a_enc = le_activity.transform([activity])[0]

years = []
preds = []

# Initial lag value (from last historical year)
temp_lag = last_value

for yr in range(last_year + 1, target_year + 1):
    # Predict in log space
    log_pred = model.predict([[yr, c_enc, a_enc, temp_lag]])[0]
    
    # Inverse transform with safe shift
    pred = np.exp(log_pred) - shift_value
    pred = max(pred, 0)  # Economic floor (cannot be negative)
    
    years.append(yr)
    preds.append(pred)
    
    # Recursive: feed prediction back as lag for next year
    temp_lag = pred

forecast_df = pd.DataFrame({
    "Year": years,
    "Predicted_Value": preds
})

# Combine historical + forecast
combined = pd.concat([
    hist[["Year", "Value"]].rename(columns={"Value": "Actual_Value"}),
    forecast_df.assign(Actual_Value=np.nan)
], ignore_index=True).sort_values("Year")

# ================================
# MAIN DASHBOARD - 3 METRICS
# ================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

current_value = last_value
projected_value = preds[-1]
growth_pct = ((projected_value - current_value) / current_value) * 100

with col1:
    st.metric(
        "📈 Current Value (2024)",
        f"${current_value:,.2f}",
        delta=None
    )

with col2:
    st.metric(
        "🎯 Projected Value (2035)",
        f"${projected_value:,.2f}",
        delta=f"{growth_pct:.2f}%"
    )

with col3:
    st.metric(
        "📊 Total Growth %",
        f"{growth_pct:.2f}%",
        delta=f"${projected_value - current_value:,.2f} USD"
    )

# ================================
# INTERACTIVE LINE CHART
# ================================
st.markdown("---")
st.subheader("📈 Historical Data vs AI Forecast (Recursive Loop)")

fig = go.Figure()

# Historical data (solid line)
hist_plot = hist.sort_values("Year")
fig.add_trace(go.Scatter(
    x=hist_plot["Year"],
    y=hist_plot["Value"],
    mode="lines+markers",
    name="Historical Data",
    line=dict(color="#0066cc", width=3),
    marker=dict(size=6)
))

# Forecast data (dashed red line)
forecast_plot = forecast_df.sort_values("Year")
fig.add_trace(go.Scatter(
    x=forecast_plot["Year"],
    y=forecast_plot["Predicted_Value"],
    mode="lines+markers",
    name="AI Forecast",
    line=dict(color="#cc0000", width=3, dash="dash"),
    marker=dict(size=6)
))

# Vertical line marking forecast start
fig.add_vline(
    x=last_year,
    line_dash="dot",
    line_color="gray",
    annotation_text="Forecast Start",
    annotation_position="top right"
)

fig.update_layout(
    title=f"{country} — Activity {activity} | 2024-2035 Projection",
    xaxis_title="Year",
    yaxis_title="Value (USD)",
    hovermode="x unified",
    template="plotly_white",
    height=500,
    font=dict(size=12, color="#003366")
)

st.plotly_chart(fig, use_container_width=True)

# ================================
# ANALYTICS TABLE
# ================================
st.markdown("---")
st.subheader("📊 Predicted Values Table (2025-2035)")

# Format forecast table
forecast_display = forecast_df.copy()
forecast_display["Year"] = forecast_display["Year"].astype(int)
forecast_display["Predicted_Value"] = forecast_display["Predicted_Value"].apply(lambda x: f"${x:,.2f}")
forecast_display = forecast_display.rename(columns={
    "Year": "Year",
    "Predicted_Value": "Forecasted USD Value"
})

st.dataframe(forecast_display, use_container_width=True, hide_index=True)

# ================================
# EXPORT OPTION
# ================================
st.markdown("---")
csv_export = forecast_df.copy()
csv_export.to_csv(f"{CHART_DIR}/forecast_{country}_{activity}_2035.csv", index=False)

st.download_button(
    label="📥 Download Forecast CSV",
    data=csv_export.to_csv(index=False),
    file_name=f"forecast_{country}_{activity}_2035.csv",
    mime="text/csv"
)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
    **💡 About This Forecast:**
    - **Recursive Loop Method**: Each year's prediction feeds back as the next year's lag input
    - **Path-Dependency**: Captures how current economic output shapes future growth
    - **Model**: ExtraTreesRegressor (300 estimators, optimized for precision)
    - **Data**: Cleaned historical data with outlier removal (IQR method)
""")

st.success("✅ Forecast generated successfully using recursive prediction loop")

# ================================
# AI-POWERED INSIGHTS & RECOMMENDATIONS
# ================================
st.markdown("---")
st.header("🤖 AI-Powered Economic Insights")

col_insights, col_recommendations = st.columns(2)

with col_insights:
    st.subheader("📊 5-Point Economic Analysis")
    if st.button("🚀 Generate AI Insights", key="insights_btn"):
        with st.spinner("🔍 Analyzing forecast data with AI..."):
            insights = generate_insights(
                country, activity, current_value, projected_value, 
                growth_pct, forecast_df
            )
            st.markdown(f"""
            <div class="insight-box">
            {insights}
            </div>
            """, unsafe_allow_html=True)

with col_recommendations:
    st.subheader("💼 Policy & Strategic Recommendations")
    if st.button("💡 Generate Recommendations", key="rec_btn"):
        with st.spinner("🔄 Generating strategic recommendations..."):
            recommendations = generate_recommendations(
                country, activity, growth_pct, projected_value
            )
            st.markdown(f"""
            <div class="insight-box">
            {recommendations}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.info("✨ These AI insights are generated using Google Gemini API and combine your forecast data with global economic trends.")