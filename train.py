import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Salary Prediction & Analytics System",
    page_icon="💼",
    layout="wide"
)

# ──────────────────────────────────────────────
# COUNTRY CODE → FULL NAME
# ──────────────────────────────────────────────
COUNTRY_NAMES = {
    "AE": "United Arab Emirates","AR": "Argentina","AT": "Austria","AU": "Australia",
    "BE": "Belgium","BG": "Bulgaria","BO": "Bolivia","BR": "Brazil","CA": "Canada",
    "CH": "Switzerland","CL": "Chile","CN": "China","CO": "Colombia","CZ": "Czech Republic",
    "DE": "Germany","DK": "Denmark","DZ": "Algeria","EE": "Estonia","ES": "Spain",
    "FR": "France","GB": "United Kingdom","GR": "Greece","HK": "Hong Kong",
    "HN": "Honduras","HR": "Croatia","HU": "Hungary","IE": "Ireland","IN": "India",
    "IQ": "Iraq","IR": "Iran","IT": "Italy","JE": "Jersey","JP": "Japan",
    "KE": "Kenya","LU": "Luxembourg","MD": "Moldova","MT": "Malta","MX": "Mexico",
    "MY": "Malaysia","NG": "Nigeria","NL": "Netherlands","NZ": "New Zealand",
    "PH": "Philippines","PK": "Pakistan","PL": "Poland","PR": "Puerto Rico",
    "PT": "Portugal","RO": "Romania","RS": "Serbia","RU": "Russia",
    "SG": "Singapore","SI": "Slovenia","TN": "Tunisia","TR": "Turkey",
    "UA": "Ukraine","US": "United States","VN": "Vietnam","ZA": "South Africa"
}

# ──────────────────────────────────────────────
# LOAD TRAINED MODEL PACKAGE
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("salary_model.pkl"):
        return None
    return joblib.load("salary_model.pkl")

model_data = load_model()

if model_data is None:
    st.error("salary_model.pkl not found. Run train.py first.")
    st.stop()

classifier = model_data["classifier"]
regressor = model_data["regressor"]
classifier_name = model_data["classifier_name"]
regressor_name = model_data["regressor_name"]
scaler = model_data["scaler"]
encoders = model_data["encoders"]
target_encoder = model_data["target_encoder"]
col_index_map = model_data["col_index_map"]
company_scale_map = model_data["company_scale_map"]
exp_tier_map = model_data["exp_tier_map"]
clf_results = model_data["clf_results"]
reg_results = model_data["reg_results"]

# ──────────────────────────────────────────────
# LOAD DATA FOR DROPDOWNS
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in ["employees.csv", "data/employees.csv", "Data/employees.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

df = load_data()
if df is None:
    st.error("employees.csv not found.")
    st.stop()

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("💼 AI Salary Intelligence System")
st.markdown("Dual Model Architecture: Classification + Regression")
st.markdown("---")

# ──────────────────────────────────────────────
# SIDEBAR INPUTS
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Employee Profile")

    work_year = st.slider("Work Year", 2020, 2026, 2024)

    # ✅ FULL EXPERIENCE LEVEL WORDS
    experience_map = {
        "Entry Level": "EN",
        "Mid Level": "MI",
        "Senior Level": "SE",
        "Executive Level": "EX"
    }
    selected_experience = st.selectbox("Experience Level", list(experience_map.keys()))
    experience_level = experience_map[selected_experience]

    # ✅ FULL EMPLOYMENT TYPE WORDS
    employment_map = {
        "Full Time": "FT",
        "Part Time": "PT",
        "Contract": "CT",
        "Freelance": "FL"
    }
    selected_employment = st.selectbox("Employment Type", list(employment_map.keys()))
    employment_type = employment_map[selected_employment]

    job_title = st.selectbox(
        "Job Title",
        df["job_title"].value_counts().head(30).index.tolist()
    )

    # ✅ FULL COUNTRY NAMES
    residence_codes = sorted(df["employee_residence"].unique())
    residence_display = [COUNTRY_NAMES.get(code, code) for code in residence_codes]
    selected_residence_name = st.selectbox("Employee Residence", residence_display)
    employee_residence = next(code for code, name in COUNTRY_NAMES.items() if name == selected_residence_name)

    location_codes = sorted(df["company_location"].unique())
    location_display = [COUNTRY_NAMES.get(code, code) for code in location_codes]
    selected_location_name = st.selectbox("Company Location", location_display)
    company_location = next(code for code, name in COUNTRY_NAMES.items() if name == selected_location_name)

    # ✅ FULL COMPANY SIZE WORDS
    company_size_map = {
        "Small Company": "S",
        "Medium Company": "M",
        "Large Company": "L"
    }
    selected_company_size = st.selectbox("Company Size", list(company_size_map.keys()))
    company_size = company_size_map[selected_company_size]

    # ✅ REMOTE LABELS
    remote_map = {
        "On-site (0%)": 0,
        "Hybrid (50%)": 50,
        "Fully Remote (100%)": 100
    }
    selected_remote = st.selectbox("Work Mode", list(remote_map.keys()))
    remote_ratio = remote_map[selected_remote]

    predict_button = st.button("Predict Salary", width="stretch")

# ──────────────────────────────────────────────
# HELPER FUNCTIONS (MATCH TRAIN.PY)
# ──────────────────────────────────────────────
def seniority_score(title):
    title = str(title).lower()
    if any(w in title for w in ["director", "head", "chief", "vp", "president"]):
        return 6
    elif any(w in title for w in ["lead", "principal", "staff", "manager"]):
        return 5
    elif "senior" in title:
        return 4
    elif any(w in title for w in ["junior", "intern"]):
        return 2
    return 3

def market_demand(title):
    title = str(title).lower()
    high = ["machine learning", "data scientist", "ai", "ml"]
    low = ["support", "assistant"]
    for h in high:
        if h in title:
            return 3
    for l in low:
        if l in title:
            return 1
    return 2

# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
if predict_button:

    input_data = np.array([[ 
        work_year - 2020,
        encoders["experience_level"].transform([experience_level])[0],
        encoders["employment_type"].transform([employment_type])[0],
        encoders["job_title"].transform([job_title])[0],
        encoders["employee_residence"].transform([employee_residence])[0],
        encoders["company_location"].transform([company_location])[0],
        encoders["company_size"].transform([company_size])[0],
        encoders["experience_tier"].transform([exp_tier_map.get(experience_level, "Mid")])[0],
        remote_ratio,
        remote_ratio / 100,
        1 if remote_ratio == 100 else 0,
        col_index_map.get(company_location, 50),
        seniority_score(job_title),
        market_demand(job_title),
        company_scale_map.get(company_size, 2)
    ]])

    input_scaled = scaler.transform(input_data)

    band_idx = classifier.predict(input_scaled)[0]
    band_pred = target_encoder.inverse_transform([band_idx])[0]

    probas = classifier.predict_proba(input_scaled)[0]
    class_names = target_encoder.classes_

    salary_log = regressor.predict(input_scaled)[0]
    salary_pred = np.expm1(salary_log)

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Salary Band", band_pred)
    col2.metric("Estimated Annual Salary (USD)", f"${salary_pred:,.0f}")

    # Ordered Confidence Graph
    band_order = ["Low", "Medium", "High", "Very High"]
    ordered_probs = []
    ordered_names = []

    for band in band_order:
        if band in class_names:
            idx = list(class_names).index(band)
            ordered_names.append(band)
            ordered_probs.append(probas[idx] * 100)

    fig = go.Figure(go.Bar(
        x=ordered_probs,
        y=ordered_names,
        orientation="h"
    ))

    fig.update_layout(
        title="Classification Confidence (%)",
        height=350
    )

    st.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────
# MODEL PERFORMANCE DASHBOARD
# ──────────────────────────────────────────────
st.markdown("---")
st.header("Model Performance Comparison")

tab1, tab2 = st.tabs(["Classification Models", "Regression Models"])

with tab1:
    clf_df = pd.DataFrame(clf_results).T
    st.dataframe(clf_df, width="stretch")

with tab2:
    reg_df = pd.DataFrame(reg_results).T
    st.dataframe(reg_df, width="stretch")

st.markdown("---")
st.markdown(
    f"✅ Best Classifier: **{classifier_name}**  |  ✅ Best Regressor: **{regressor_name}**"
)
