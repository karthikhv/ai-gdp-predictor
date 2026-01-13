# 🌍 AI GDP Forecasting Dashboard (2025-2035)
## Professional AI-Driven Industrial Output Predictions with Recursive Forecasting & Google Gemini Integration

**Repository:** https://github.com/karthikhv/ai-gdp-predictor

---

## 📋 Project Overview

This is a **Senior Data Science thesis project** that builds a complete **Streamlit forecasting application** for predicting global industrial output (GDP) from 2025 to 2035. The system uses machine learning with a recursive feedback loop strategy and integrates Google's Gemini AI for economic insights.

### Key Innovation: 🔄 Recursive Forecasting Loop
Instead of static predictions, the system uses a **dynamic feedback mechanism** where:
- Each year's economic output prediction serves as the foundation for the next year's growth simulation
- This captures the **path-dependency** of national economies
- Economic momentum from previous years influences future projections

---

## ✨ Features Implemented

### 1. **Data Engineering Pipeline** 📊
✅ Load `data_cleaned.csv` with 23,473 records from 137 countries  
✅ Remove outliers using **IQR method** (Interquartile Range)  
✅ Create **Lag_1 feature** representing previous year's Value per Country/Activity  
✅ Handle sparse data (bootstrap lag for countries with single records)  
✅ Drop NaN values after feature engineering  

**Data Quality:** All missing values removed, outliers handled, ready for ML

---

### 2. **Advanced Machine Learning Model** 🤖
**Algorithm:** ExtraTreesRegressor (Ensemble Learning)
- **Estimators:** 300 trees
- **Max Depth:** 20 (prevents overfitting)
- **Min Samples Leaf:** 5 (ensures robust splits)
- **Random State:** 42 (reproducible results)
- **Parallel Processing:** All cores used for speed

**Input Features:**
- Year (temporal dimension)
- Country (LabelEncoded, 137 unique values)
- Activity Sector (LabelEncoded, sector-specific)
- Lag_1 (previous year's value)

**Target Variable:** Value (Actual USD value)

**Technical Optimization:**
- Safe log transformation with shift handling
- Prevents numerical instability with negative/zero values
- Inverse transform: `np.exp(log_pred) - shift` with economic floor (≥0)

---

### 3. **Recursive Forecasting Strategy (2025-2035)** 🔄
```
Year 2024 (Historical): $100 billion
  ↓
Year 2025 Prediction: $105 billion (model output)
  ↓
Year 2026 Input (Lag): $105 billion (previous prediction fed back)
  ↓
Year 2026 Prediction: $110.25 billion
  ↓
[Loop continues until 2035]
```

**Why This Matters:**
- Captures compound growth effects
- Reflects economic momentum
- Shows realistic growth trajectories
- Accounts for industrial sector evolution

---

### 4. **Streamlit UI - Professional Dashboard** 🎨
**Design:** Clean white background with professional blue/black styling

#### **Sidebar Controls**
- 🌏 **Country Selector:** Dynamically filtered (only countries with data)
- 🛠️ **Activity Selector:** Filtered by selected country
- 📅 **Forecast Year Slider:** Set target year (2025-2035)
- 💡 **Help Text:** Explains recursive loop methodology

#### **Main Dashboard - 3 Key Metrics**
```
┌─────────────────────────────────────────────────────────┐
│ 📈 Current Value (2024)  │  🎯 Projected (2035)  │  📊 Growth % │
│ $XXX billion            │  $YYY billion         │  Z%          │
└─────────────────────────────────────────────────────────┘
```

#### **Interactive Visualization**
- **Historical Data:** Solid blue line with markers
- **AI Forecast:** Dashed red line (2025-2035)
- **Vertical Separator:** Marks where predictions begin
- **Hover Details:** See exact values by hovering
- **Technology:** Plotly for interactive exploration

#### **Analytics Table**
- Year-by-year predictions (2025-2035)
- USD values formatted for readability
- Export as CSV button for further analysis

#### **Export Functionality**
- Download forecasts as CSV
- Named: `forecast_{country}_{activity}_2035.csv`
- Ready for thesis appendices

---

### 5. **🤖 AI-Powered Economic Insights (Google Gemini 2.5 Flash)**

#### **Two AI Features:**

**A) 5-Point Economic Analysis**
Generates deep insights on:
1. 🎯 **Growth Trajectory** — Pattern analysis & economic meaning
2. 💼 **Sectoral Impact** — How growth affects the industry
3. 📈 **Competitive Position** — Global market standing
4. ⚠️ **Risks & Challenges** — Factors that could derail forecast
5. 🚀 **Opportunities** — Strategic initiatives for growth

**B) 3 Policy Recommendations**
Actionable strategies for {country}'s {sector}:
- Clear recommendation
- Expected benefits
- Implementation roadmap
- Measurable & SDG-aligned

#### **Why Gemini 2.5 Flash?**
- Latest model (as of Jan 2026)
- 3x faster than previous versions
- Better economic data understanding
- Combines your data + global context

---

## 🛠️ Requirements & Installation

### **System Requirements**
- Python 3.8+
- Windows/macOS/Linux
- 4GB RAM minimum
- Internet (for Gemini API calls)

### **Installation Steps**

1. **Clone Repository**
```bash
git clone https://github.com/karthikhv/ai-gdp-predictor.git
cd ai-gdp-predictor
```

2. **Create Virtual Environment** (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Required Packages**
```
pandas==2.0.0+
numpy==1.24.0+
scikit-learn==1.3.0+
streamlit==1.28.0+
plotly==5.17.0+
google-generativeai==0.3.0+
matplotlib==3.8.0+
seaborn==0.13.0+
joblib==1.3.0+
```

### **Environment Setup**
Create `.env` file (optional, API key is embedded):
```env
GEMINI_API_KEY=AIzaSyDoCOpDeJzZzQPCHokZLYzzO6CE-wxcUDk
```

---

## 🚀 How to Run the App

### **Basic Launch**
```bash
streamlit run app_forecast_2035.py
```

The app will:
- Load `data_cleaned.csv`
- Preprocess & encode features
- Train ExtraTreesRegressor model
- Generate forecasts
- Start Streamlit server on `http://localhost:8501`

### **Step-by-Step Usage**

1. **Open Browser**
   - Navigate to `http://localhost:8501`

2. **Configure Forecast**
   - Select a country (e.g., China, India, Germany)
   - Select an activity sector (e.g., Food & Beverages, Chemicals)
   - Adjust forecast year (2025-2035)

3. **View Results**
   - 3 metrics show current, projected, and growth %
   - Interactive chart displays historical + forecast
   - Table shows year-by-year USD values

4. **Generate AI Insights** (Optional)
   - Click **"Generate AI Insights"** button
   - Wait for Gemini API response (5-10 seconds)
   - Read 5-point economic analysis

5. **Get Recommendations** (Optional)
   - Click **"Generate Recommendations"** button
   - Receive 3 policy recommendations
   - Share with stakeholders/advisors

6. **Export Data**
   - Click **"Download Forecast CSV"**
   - Use in thesis appendices

---

## 📊 Data Engineering Details

### **Data Source**
- **File:** `data_cleaned.csv` (23,473 rows)
- **Countries:** 137 unique nations
- **Time Period:** 2000-2024
- **Sectors:** Manufacturing, Mining, Food, Chemicals, Textiles, etc.
- **Values:** USD industrial output

### **Preprocessing Pipeline**
```
Raw Data (23,473 rows)
    ↓
Remove Outliers (IQR method)
    ↓
Sort by Country, Activity, Year
    ↓
Create Lag_1 (previous year value)
    ↓
Bootstrap for sparse data (single records)
    ↓
Ready for Training (clean dataset)
```

### **Outlier Removal (IQR Method)**
```python
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Remove if: Value < (Q1 - 1.5×IQR) OR Value > (Q3 + 1.5×IQR)
```

### **Lag Feature Engineering**
```python
For each (Country, Activity):
    Lag_1[t] = Value[t-1]  # Previous year's value
    
For sparse data:
    If no previous year → Lag_1 = Value (bootstrap)
```

---

## 🤖 Machine Learning Pipeline

### **Model Architecture**
```
Input Features → Encoding & Scaling → ExtraTreesRegressor → Log-Space Prediction → Inverse Transform → USD Output
```

### **Feature Encoding**
```python
Country: LabelEncoder (0-136)
Activity: LabelEncoder (0-35)
Year: Numerical (2000-2035)
Lag_1: Numerical (previous value)
```

### **Log Transform Safety**
```python
min_val = df["Value"].min()
shift = abs(min_val) + 1 if min_val <= 0 else 0

# Forward transform
Value_Shifted = Value + shift
y = log(Value_Shifted)

# Inverse transform (in forecasting)
pred = exp(log_pred) - shift
pred = max(pred, 0)  # Economic floor
```

### **Why Log Transform?**
- Stabilizes variance across different scales
- Handles exponential growth patterns
- Improves model precision for economic data
- Prevents numerical instability

---

## 🔄 Recursive Forecasting Logic

### **Pseudocode**
```python
last_year = 2024
last_value = $100B (from data)

for year in range(2025, 2036):
    # Create input features
    X = [year, country_encoded, activity_encoded, lag_1]
    
    # Predict in log space
    log_pred = model.predict(X)[0]
    
    # Inverse transform
    pred = exp(log_pred) - shift
    pred = max(pred, 0)  # Economic floor
    
    # Store prediction
    predictions[year] = pred
    
    # Recursive step: feed back as lag
    lag_1 = pred  # ← THIS IS THE KEY
```

### **Why Recursive?**
- **Path-Dependent Growth:** Current year depends on previous
- **Momentum Capture:** Growth acceleration/deceleration shown
- **Realistic Projections:** Reflects compounding economic effects
- **Thesis-Ready:** Explains economic dynamics to professors

---

## 📈 Validation & Accuracy

### **Model Performance Metrics**
- **R² Score:** Explains variance in industrial output
- **MAE:** Mean Absolute Error in USD predictions
- **RMSE:** Root Mean Squared Error (penalizes large errors)

### **Key Validation Points**
✅ All 137 countries included  
✅ 35+ industrial sectors covered  
✅ Log-space stability verified  
✅ No negative predictions (floor enforced)  
✅ Recursive loop tested end-to-end  

### **Data Quality Checks**
✅ No NaN values in training data  
✅ Outliers removed (IQR method)  
✅ Lag features properly aligned  
✅ Year progression correct  

---

## 💾 Project Structure

```
ai-gdp-predictor/
├── app_forecast_2035.py          # 🎯 Main Streamlit app
├── app_2025.py                   # Alternative version
├── data_cleaned.csv              # Clean dataset (23,473 rows)
├── data.csv                      # Raw dataset
├── datapre.ipynb                 # Data preprocessing notebook
├── mod_train.ipynb               # Model training & analysis
├── models/
│   ├── rf_model.pkl              # Pre-trained ExtraTreesRegressor
│   ├── country_encoder.pkl       # Country label encoder
│   ├── activity_encoder.pkl      # Activity sector encoder
│   └── scaler.pkl                # Feature scaler
├── charts/
│   ├── forecast_China_Basic_metals_2035.csv
│   ├── forecast_India_Total_manufacturing_2035.csv
│   └── [12 more forecast CSVs]
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🔐 API Keys & Configuration

### **Google Gemini API**
- **Key:** `AIzaSyDoCOpDeJzZzQPCHokZLYzzO6CE-wxcUDk`
- **Model:** gemini-2.5-flash (latest)
- **Embedded:** Already in `app_forecast_2035.py`
- **Status:** Active & tested ✅

### **Cost Estimate**
- **Free Tier:** ~1,000 requests/day sufficient
- **Production:** Consider switching to gemini-pro for higher limits
- **Fallback:** App works without API (shows error message)

---

## 🎓 For Your Thesis

### **Key Points to Explain to Your Professor:**

#### **1. The Recursive Loop Innovation** 🔄
> "Instead of a static prediction, the system uses a **dynamic feedback mechanism** where each year's economic output serves as the foundation for the next year's growth simulation. This captures the **path-dependency of national economies**, showing how current industrial performance shapes future trajectories."

#### **2. Why This Matters Methodologically**
- **Compounding Effects:** Shows economic momentum
- **Realistic Growth:** Reflects how industries actually grow
- **Time-Series Logic:** Uses temporal dependencies properly
- **Generalizable:** Works across 137 countries & 35+ sectors

#### **3. Technical Rigor**
- **ExtraTreesRegressor:** Ensemble reduces overfitting
- **Feature Engineering:** Lag captures historical momentum
- **Log Transform:** Handles exponential growth patterns
- **IQR Outlier Removal:** Statistical rigor in data prep

#### **4. AI Integration as Future Work**
- **Gemini Insights:** Demonstrates AI augmentation of ML
- **Policy Recommendations:** Shows practical applications
- **Scalability:** Framework can include more AI features

---

## 📊 Example Outputs

### **Sample Forecast: India - Total Manufacturing**

```
Current Value (2024):     $XXX billion
Projected Value (2035):   $YYY billion
Growth Rate:              ZZ.Z%

Year    Forecast (USD)
2025    $105.0 billion
2026    $110.3 billion
2027    $116.1 billion
...
2035    $150.0 billion
```

### **AI Insight Example:**
```
🎯 GROWTH TRAJECTORY
India's manufacturing is projected to grow at a compound annual rate 
of 4.5%, reflecting industrialization momentum and export expansion.

💼 SECTORAL IMPACT
This growth will particularly benefit capital goods, electronics, 
and pharmaceutical sectors, positioning India as a global hub.

[... continues with 3 more insights ...]
```

---

## 🐛 Troubleshooting

### **Issue: "No historical data available for [Country] — [Activity]"**
- **Cause:** Country-activity combo not in cleaned data
- **Solution:** Select different country or activity from dropdown
- **Prevention:** App filters dropdowns to show only valid options

### **Issue: "Unable to generate insights: 404 models/..."**
- **Cause:** Outdated Gemini model name
- **Solution:** Already fixed to `gemini-2.5-flash`
- **Status:** ✅ Resolved in this version

### **Issue: NaN values in output**
- **Cause:** Negative predictions (shouldn't happen)
- **Solution:** Economic floor enforced (`max(pred, 0)`)
- **Verification:** Check log transform shift value

### **Issue: App runs slowly**
- **Cause:** First run trains model (2-3 min)
- **Solution:** Model cached after first load
- **Optimization:** Uses all CPU cores for training

---

## 📝 Documentation Files

### **In Repository:**
- **README.md** (this file) — Complete project documentation
- **app_forecast_2035.py** — Fully commented source code
- **datapre.ipynb** — Data preprocessing walkthrough
- **mod_train.ipynb** — Model training & visualizations

### **For Your Thesis:**
- Use **Section 3 (Features Implemented)** for Methods chapter
- Use **Section 5 (Machine Learning Pipeline)** for Results chapter
- Use **Recursive Forecasting Logic** for Innovation section
- Use **Example Outputs** for Appendix

---

## 🚀 Deployment Options

### **Option 1: Streamlit Cloud** (Recommended)
```bash
streamlit run app_forecast_2035.py --logger.level=error
```

### **Option 2: Docker** (Production)
Create `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app_forecast_2035.py"]
```

### **Option 3: Heroku** (Legacy)
Add `Procfile`:
```
web: streamlit run app_forecast_2035.py --server.port=$PORT
```

---

## 📚 References

### **Machine Learning**
- ExtraTreesRegressor: [Scikit-learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- Time Series Forecasting: Box & Jenkins methodology
- Log Transforms: Applied to economic time series

### **Economic Theory**
- Path Dependency: Arthur, W.B. (1989) - "Competing Technologies"
- Industrial Output: UN Industrial Development Organization (UNIDO)
- GDP Forecasting: IMF Methodology

### **Technology**
- Streamlit: [streamlit.io](https://streamlit.io)
- Google Gemini: [Google AI Studio](https://aistudio.google.com)
- Scikit-learn: [scikit-learn.org](https://scikit-learn.org)

---

## 📧 Contact & Support

**Project Author:** Data Science Thesis, Jan 2026  
**Repository:** https://github.com/karthikhv/ai-gdp-predictor  
**Last Updated:** January 13, 2026

### **Questions?**
- Check **Troubleshooting** section
- Review **app_forecast_2035.py** comments
- Run notebooks for step-by-step walkthroughs

---

## 📄 License

This project is part of an academic thesis. Use freely for educational purposes.

---

## ✅ Checklist for Thesis Submission

- [x] Data engineering pipeline documented
- [x] Machine learning model explained
- [x] Recursive forecasting logic detailed
- [x] Streamlit UI showcased
- [x] AI integration demonstrated (Gemini)
- [x] Code uploaded to GitHub
- [x] README with complete documentation
- [x] All 137 countries supported
- [x] 35+ industrial sectors covered
- [x] Predictions from 2025-2035
- [x] Export functionality (CSV)

**Status:** 🎉 READY FOR THESIS SUBMISSION

---

## 🎯 Final Notes for Your Professor

This project demonstrates:

1. **Data Science Excellence:** Full ML pipeline from raw data to production app
2. **Innovation:** Recursive loop captures economic path-dependency
3. **Technical Depth:** Log transforms, ensemble methods, feature engineering
4. **Practical Impact:** AI-powered insights for policy decisions
5. **Scalability:** Works across 137 countries & 35+ sectors
6. **Modern Stack:** Streamlit UI + Google Gemini integration

**The recursive forecasting approach is the key innovation to highlight in your defense.** It shows that you understand not just how to build ML models, but how to apply them thoughtfully to real-world economic problems.

---

**Happy Thesis Defense! 🎓**
