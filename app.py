import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER
 
st.set_page_config(
    page_title="CompensIQ",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("artifacts/model.pkl")
    scaler   = joblib.load("artifacts/scaler.pkl")
    fcols    = joblib.load("artifacts/feature_columns.pkl")
    cat_cols = joblib.load("artifacts/cat_cols.pkl")
    with open("model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, fcols, cat_cols, meta
 
try:
    model, scaler, feature_cols, cat_cols, meta = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()
 
# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
 
.main-header {
    background: linear-gradient(135deg, #0F1B5F 0%, #1565C0 50%, #0D47A1 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(21, 101, 192, 0.3);
}
.main-header h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
}
 
.salary-box {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
}
.salary-box .label {
    font-size: 0.95rem;
    opacity: 0.85;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0;
}
.salary-box .amount {
    font-size: 4rem;
    font-weight: 800;
    margin: 0.3rem 0;
    letter-spacing: -2px;
}
.salary-box .band {
    font-size: 1.1rem;
    font-weight: 600;
    background: rgba(255,255,255,0.2);
    display: inline-block;
    padding: 0.2rem 1rem;
    border-radius: 20px;
    margin: 0.3rem 0;
}
.salary-box .range {
    font-size: 0.9rem;
    opacity: 0.8;
    margin: 0.3rem 0 0 0;
}
 
.metric-card {
    background: white;
    border: 1px solid #E3F2FD;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .metric-label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1565C0;
}
 
.component-row {
    display: flex;
    align-items: center;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    background: #F8F9FA;
    border-left: 4px solid #1565C0;
}
 
.ladder-card {
    background: #F8F9FA;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
 
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0F1B5F;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E3F2FD;
}
 
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.1rem;
}
.badge-blue { background: #E3F2FD; color: #1565C0; }
.badge-green { background: #E8F5E9; color: #2E7D32; }
.badge-orange { background: #FFF3E0; color: #E65100; }
 
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #1565C0;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    color: white !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1565C0!important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
 
.sidebar-section {
    background: #F0F4FF;
    border-radius: 10px;
    padding: 0.8rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)
 
# ── Constants ─────────────────────────────────────────────────
REGION_MAP = {
    "US":"North America","CA":"North America","GB":"Europe_West",
    "DE":"Europe_West","FR":"Europe_West","NL":"Europe_West",
    "IN":"Asia_Em","AU":"Asia_Dev","SG":"Asia_Dev",
    "BR":"LatAm","ZA":"Africa","AR":"LatAm",
    "ES":"Europe_South","PL":"Europe_East","LT":"Europe_East",
}
COUNTRY_GEO = {
    "US":1.000,"CA":0.780,"GB":0.700,"DE":0.680,"FR":0.600,
    "NL":0.730,"IN":0.180,"AU":0.740,"SG":0.680,"BR":0.250,
    "ZA":0.250,"AR":0.180,"ES":0.460,"PL":0.320,"LT":0.240,
}
COUNTRY_NAMES = {
    "US":"United States","CA":"Canada","GB":"United Kingdom",
    "DE":"Germany","FR":"France","NL":"Netherlands",
    "IN":"India","AU":"Australia","SG":"Singapore",
    "BR":"Brazil","ZA":"South Africa","AR":"Argentina",
    "ES":"Spain","PL":"Poland","LT":"Lithuania",
}
BASE_SAL = {
    "ML Engineer":128000,"Data Scientist":118000,
    "Manager":140000,"Architect":135000,
    "Data Engineer":115000,"Analyst":88000,"Other":100000,
}
TIER_MULT = {
    "Startup":0.82,"Mid-tier":1.00,
    "Enterprise":1.12,"Big Tech":1.45
}
EDU_M = {
    "Bachelor":1.00,"Master":1.08,"PhD":1.18,"MBA":1.10
}
IND_M = {
    "Tech":1.00,"Finance":1.22,"Research":1.05,
    "Consulting":1.08,"Other_Industry":0.90
}
REMOTE_M = {
    0:1.02,10:1.01,20:1.01,25:1.00,30:1.00,
    40:0.99,50:0.99,60:0.98,70:0.97,
    75:0.97,80:0.96,90:0.96,100:0.95,
}
MARKET = {
    2020:0.91,2021:0.94,2022:1.06,
    2023:1.09,2024:1.00,2025:0.98,2026:1.01
}
EXP_BOUNDS = {
    "EN":(0,2,1),"MI":(3,6,4),
    "SE":(7,12,9),"EX":(13,22,16)
}
SIZE_LABELS = {
    "S":"Small — Startup (<50 employees)",
    "M":"Medium — Mid-tier (50-999 employees)",
    "L":"Large — Enterprise / Big Tech (1000+)",
}
 
def get_tier(size, job_family):
    if size == "S": return "Startup"
    if size == "M": return "Mid-tier"
    if job_family in ["ML Engineer","Data Scientist"]: return "Big Tech"
    return "Enterprise"
 
def exp_multiplier(years):
    m = 1.0
    for yr in range(1, int(years)+1):
        if yr<=2: m*=1.060
        elif yr<=5: m*=1.045
        elif yr<=8: m*=1.030
        elif yr<=12: m*=1.018
        else: m*=1.010
    return m
 
def predict_salary(exp_level, years_exp, job_family, company_size,
                   company_tier, country, city_tier, education,
                   industry, remote_pct, work_year):
    region   = REGION_MAP.get(country, "Other")
    geo_m    = COUNTRY_GEO.get(country, 0.300)
    is_us    = 1 if country == "US" else 0
    base     = BASE_SAL.get(job_family, 100000)
    comp_m   = TIER_MULT.get(company_tier, 1.0)
    edu_m    = EDU_M.get(education, 1.0)
    ind_m    = IND_M.get(industry, 1.0)
    rem_m    = REMOTE_M.get(remote_pct, 0.97)
    yr_m     = MARKET.get(work_year, 1.0)
    exp_mult = exp_multiplier(years_exp)
    skill_m  = 1.0 + (years_exp * 0.006)
    snr      = 1 if years_exp >= 7 else 0
    signal   = base*exp_mult*comp_m*geo_m*edu_m*ind_m*rem_m*yr_m
    log_sig  = np.log1p(signal)
    yrs_2020 = work_year - 2020
    mkt_cyc  = MARKET.get(work_year, 1.0)
    is_2026f = 1 if work_year == 2026 else 0
 
    row = {
        "experience_level":exp_level,"job_family":job_family,
        "company_size":company_size,"company_tier":company_tier,
        "education_level":education,"industry":industry,
        "city_tier":city_tier,"company_region":region,
        "employee_region":region,
        "remote_type":("onsite" if remote_pct<30
                       else "hybrid" if remote_pct<80 else "remote"),
        "years_of_experience":years_exp,"seniority_flag":snr,
        "is_us":is_us,"geo_multiplier":geo_m,
        "remote_fraction":remote_pct/100.0,
        "years_since_2020":yrs_2020,"market_cycle":mkt_cyc,
        "is_2026":is_2026f,"base_salary_index":base,
        "company_multiplier":comp_m,"edu_multiplier":edu_m,
        "industry_multiplier":ind_m,"remote_multiplier":rem_m,
        "experience_multiplier":exp_mult,"skills_multiplier":skill_m,
        "expected_salary_signal":signal,"log_expected_signal":log_sig,
        "exp_x_geo":years_exp*geo_m,"exp_x_company":years_exp*comp_m,
        "exp_x_edu":years_exp*edu_m,"seniority_x_exp":snr*years_exp,
        "geo_x_tier":geo_m*comp_m,"signal_x_market":signal*mkt_cyc,
    }
    idf  = pd.DataFrame([row])
    ienc = pd.get_dummies(idf, columns=cat_cols, drop_first=True)
    ienc = ienc.reindex(columns=feature_cols, fill_value=0)
    pred = float(np.expm1(model.predict(ienc)[0]))
    return max(pred, 15000), signal, base, comp_m, geo_m, edu_m, ind_m, exp_mult
 
# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>CompensIQ</h1>
    <p>Data Science Salary Intelligence System</p>
</div>
""", unsafe_allow_html=True)
 
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Model", "HistGradBoost")
col_m2.metric("R² Score", f"{meta['r2']:.4f}")
col_m3.metric("CV R²",    f"{meta['cv_r2']:.4f}")
col_m4.metric("MAE",      f"${meta['mae']:,.0f}")
col_m5.metric("MAPE",     f"{meta['mape']:.1f}%")
 
st.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Jaswanth--ak%2Fcompensiq-black?logo=github&style=flat-square)]"
    "(https://github.com/Jaswanth-ak/compensiq)",
    unsafe_allow_html=True
)
st.divider()
 
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Candidate Profile")
    candidate_name = st.text_input("Candidate Name", "Your Name")
 
    st.markdown("---")
    st.markdown("### Experience")
    exp_level = st.selectbox(
        "Experience Level",
        ["EN","MI","SE","EX"],
        format_func=lambda x: {
            "EN":"Entry Level  (0-2 yr)",
            "MI":"Mid Level    (3-6 yr)",
            "SE":"Senior Level (7-12 yr)",
            "EX":"Executive    (13+ yr)"
        }[x], index=2
    )
    exp_lo, exp_hi, exp_def = EXP_BOUNDS[exp_level]
    years_exp = st.slider(
        "Years of Experience",
        min_value=exp_lo, max_value=exp_hi,
        value=min(exp_def, exp_hi)
    )
    st.caption(f"Valid range for this level: {exp_lo}–{exp_hi} years")
 
    st.markdown("---")
    st.markdown("### Role")
    job_family = st.selectbox(
        "Job Family",
        ["Data Scientist","ML Engineer","Data Engineer",
         "Analyst","Manager","Architect","Other"]
    )
    education = st.selectbox(
        "Education Level",
        ["Bachelor","Master","PhD","MBA"]
    )
 
    st.markdown("---")
    st.markdown("### Company")
    company_size = st.selectbox(
        "Company Size",
        ["S","M","L"],
        format_func=lambda x: SIZE_LABELS[x],
        index=1
    )
    company_tier = get_tier(company_size, job_family)
    st.info(f"Tier: **{company_tier}** | Multiplier: ×{TIER_MULT[company_tier]:.2f}")
 
    st.markdown("---")
    st.markdown("### Location")
    country = st.selectbox(
        "Country",
        list(COUNTRY_NAMES.keys()),
        format_func=lambda x: COUNTRY_NAMES[x]
    )
if country != "US":
    city_tier = "International"
    st.info("City Tier: International (auto-set for non-US)")
else:
    city_tier = st.selectbox(
        "City Tier (US only)",
        ["Tier1_Metro","Tier2_City","Tier3_Other"],
        format_func=lambda x: {
            "Tier1_Metro":"Tier 1 — SF / NYC / Seattle (+18%)",
            "Tier2_City":"Tier 2 — Austin / Boston / Chicago (+5%)",
            "Tier3_Other":"Tier 3 — Rest of US (-5%)",
        }[x]
    )
    st.caption("Only shown for US — other countries use geo multiplier")
 
    st.markdown("---")
    st.markdown("### Work Setup")
    industry = st.selectbox(
        "Industry",
        ["Tech","Finance","Research","Consulting","Other_Industry"],
        format_func=lambda x: {
            "Tech":"Tech",
            "Finance":"Finance  (+22% premium)",
            "Research":"Research (+5% premium)",
            "Consulting":"Consulting (+8% premium)",
            "Other_Industry":"Other Industry (-10%)"
        }[x]
    )
    remote_pct = st.select_slider(
        "Remote Work %",
        options=[0,10,20,25,30,40,50,60,70,75,80,90,100],
        value=0
    )
    work_year = st.selectbox(
        "Prediction Year",
        [2023,2024,2025,2026], index=2
    )
 
# ── Compute prediction ────────────────────────────────────────
prediction, signal, base, comp_m, geo_m, edu_m, ind_m, exp_mult = predict_salary(
    exp_level, years_exp, job_family, company_size,
    company_tier, country, city_tier, education,
    industry, remote_pct, work_year
)
 
mae     = meta["mae"]
low     = max(prediction - mae, 15000)
high    = prediction + mae
monthly = prediction / 12
weekly  = prediction / 52
hourly  = prediction / 2080
 
if prediction < 60000:    band = "Entry Market"
elif prediction < 100000: band = "Below Midpoint"
elif prediction < 150000: band = "At Market Rate"
elif prediction < 200000: band = "Above Midpoint"
elif prediction < 280000: band = "Top Quartile"
else:                     band = "Top Percentile"
 
# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Salary Breakdown",
    "Model Analytics",
    "Download Report"
])
 
# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════
with tab1:
 
    st.markdown(f"""
    <div class="salary-box">
        <p class="label">Estimated Annual Base Salary</p>
        <p class="amount">${prediction:,.0f}</p>
        <span class="band">{band}</span>
        <p class="range">Range: ${low:,.0f} &ndash; ${high:,.0f} &nbsp;|&nbsp; ± MAE ${mae:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly",  f"${monthly:,.0f}")
    c2.metric("Weekly",   f"${weekly:,.0f}")
    c3.metric("Hourly",   f"${hourly:,.0f}")
    c4.metric("MAE ±",    f"${mae:,.0f}")
 
    st.divider()
    st.markdown("### Salary Comparison Ladders")
    st.caption("See how your profile compares across experience levels and company sizes")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("**By Experience Level**")
        st.caption("Same role, company, and location — only experience changes")
        for lvl, yrs, label in [
            ("EN",1,"Entry"),("MI",4,"Mid"),
            ("SE",9,"Senior"),("EX",16,"Exec")
        ]:
            p_, *_ = predict_salary(
                lvl, yrs, job_family, company_size,
                company_tier, country, city_tier,
                education, industry, remote_pct, work_year
            )
            is_you = lvl == exp_level
            diff   = p_ - prediction
            diff_s = f"+${diff:,.0f}" if diff > 0 else f"-${abs(diff):,.0f}" if diff < 0 else "YOU"
            color  = "#2E7D32" if is_you else "#555"
            bg     = "#E8F5E9" if is_you else "#F8F9FA"
            st.markdown(
                f"""<div style="background:{bg};border-radius:8px;padding:0.7rem 1rem;
                margin:0.3rem 0;display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:{'700' if is_you else '500'};color:{color};">
                {label} {'← YOU' if is_you else ''}</span>
                <span style="font-weight:700;color:{color};">${p_:,.0f}</span>
                </div>""",
                unsafe_allow_html=True
            )
 
    with col2:
        st.markdown("**By Company Size**")
        st.caption("Same role and experience — only company size changes")
        for sz_, label_ in [
            ("S","Small — Startup"),
            ("M","Medium — Mid-tier"),
            ("L","Large — Enterprise/BigTech"),
        ]:
            tr_ = get_tier(sz_, job_family)
            p_, *_ = predict_salary(
                exp_level, years_exp, job_family, sz_,
                tr_, country, city_tier,
                education, industry, remote_pct, work_year
            )
            is_you = sz_ == company_size
            color  = "#1565C0" if is_you else "#555"
            bg     = "#E3F2FD" if is_you else "#F8F9FA"
            st.markdown(
                f"""<div style="background:{bg};border-radius:8px;padding:0.7rem 1rem;
                margin:0.3rem 0;display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:{'700' if is_you else '500'};color:{color};">
                {label_} {'← YOU' if is_you else ''}</span>
                <span style="font-weight:700;color:{color};">${p_:,.0f}</span>
                </div>""",
                unsafe_allow_html=True
            )
 
# ════════════════════════════════════════════════════════════════
# TAB 2 — BREAKDOWN
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Salary Component Breakdown")
    st.caption("Every dollar of the prediction explained by a real economic factor")
 
    exp_impact  = base * (exp_mult - 1)
    comp_impact = base * exp_mult * (comp_m - 1)
    geo_impact  = base * exp_mult * comp_m * (geo_m - 1)
    edu_impact  = base * exp_mult * comp_m * geo_m * (edu_m - 1)
    ind_impact  = base * exp_mult * comp_m * geo_m * edu_m * (ind_m - 1)
    ml_adj      = prediction - signal
 
    components = [
        ("Base Salary",      f"${base:,.0f}",
         f"{job_family} US market benchmark",         "#1565C0"),
        ("Experience Curve", f"+${exp_impact:,.0f}",
         f"{years_exp} years — non-linear BLS growth curve, ×{exp_mult:.3f}", "#7B1FA2"),
        ("Company Size",     f"${comp_impact:+,.0f}",
         f"{SIZE_LABELS[company_size].split('—')[0].strip()} ({company_tier}), ×{comp_m:.2f}", "#E65100"),
        ("Geography",        f"${geo_impact:+,.0f}",
         f"{COUNTRY_NAMES.get(country, country)} market rate, ×{geo_m:.2f}", "#00695C"),
        ("Education",        f"+${edu_impact:,.0f}",
         f"{education} degree premium, ×{edu_m:.2f}", "#1565C0"),
        ("Industry",         f"${ind_impact:+,.0f}",
         f"{industry} sector adjustment, ×{ind_m:.2f}", "#6A1B9A"),
        ("Economic Signal",  f"${signal:,.0f}",
         "Formula-based estimate before ML correction", "#37474F"),
        ("ML Correction",    f"${ml_adj:+,.0f}",
         "Pattern adjustment from 49,495 salary records", "#BF360C"),
    ]
 
    for label, value, desc, color in components:
        st.markdown(
            f"""<div style="background:#F8F9FA;border-radius:10px;padding:0.85rem 1.2rem;
            margin:0.4rem 0;border-left:4px solid {color};
            display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-weight:600;color:#222;font-size:0.95rem;">{label}</div>
                <div style="color:#666;font-size:0.82rem;margin-top:2px;">{desc}</div>
            </div>
            <div style="font-weight:800;font-size:1.1rem;color:{color};white-space:nowrap;margin-left:1rem;">
                {value}
            </div>
            </div>""",
            unsafe_allow_html=True
        )
 
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#1B5E20,#2E7D32);
        border-radius:12px;padding:1.2rem 1.5rem;margin:1rem 0;
        display:flex;justify-content:space-between;align-items:center;">
        <span style="color:white;font-size:1.1rem;font-weight:600;">
        Final Prediction — {band}</span>
        <span style="color:white;font-size:1.8rem;font-weight:800;">
        ${prediction:,.0f}</span>
        </div>""",
        unsafe_allow_html=True
    )
 
    st.divider()
    st.markdown("### Negotiation Strategy")
    n1, n2, n3 = st.columns(3)
    n1.metric("Conservative Ask", f"${prediction*0.90:,.0f}", "-10%",
              delta_color="inverse")
    n2.metric("Market Rate",      f"${prediction:,.0f}",      "Your profile")
    n3.metric("Stretch Ask",      f"${prediction*1.12:,.0f}", "+12%")
 
    st.info(
        "Conservative = use when you have no competing offers. "
        "Market Rate = use when you match the profile closely. "
        "Stretch = use when you have competing offers or strong track record."
    )
 
# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL ANALYTICS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Performance")
 
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Algorithm", meta["best_model"])
    m2.metric("R² Score",  f"{meta['r2']:.4f}")
    m3.metric("CV R²",     f"{meta['cv_r2']:.4f} ± {meta['cv_std']:.4f}")
    m4.metric("MAE",       f"${meta['mae']:,.0f}")
    m5.metric("MAPE",      f"{meta['mape']:.1f}%")
 
    st.divider()
    st.markdown("### All Models Compared")
    if "all_models" in meta:
        rows = []
        for name, m in meta["all_models"].items():
            rows.append({
                "Model": name,
                "R²": m["r2"],
                "MAE": f"${m['mae']:,.0f}",
                "MAPE": f"{m['mape']:.1f}%",
                "Status": "Best" if name == meta["best_model"] else ""
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )
 
    st.divider()
    st.markdown("### Diagnostic Charts")
    chart_files = [
        ("charts/07_salary_distributions.png", "Salary Distributions — All Dimensions"),
        ("charts/03_diagnostics.png",           "Actual vs Predicted + Residuals"),
        ("charts/04_band_performance.png",      "Performance by Salary Band"),
        ("charts/05_feature_importance.png",    "Feature Importance"),
        ("charts/06_model_comparison.png",      "Model Comparison"),
        ("charts/02_correlation.png",           "Feature Correlation Matrix"),
    ]
    for img, title in chart_files:
        if os.path.exists(img):
            st.markdown(f"**{title}**")
            st.image(img, use_container_width=True)
            st.divider()
 
# ════════════════════════════════════════════════════════════════
# TAB 4 — DOWNLOAD REPORT
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Candidate Eligibility Report")
    st.markdown(
        "Generate a professional PDF report with full salary breakdown, "
        "eligibility assessment, strengths, development areas, and negotiation guidance."
    )
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style="background:#F0F4FF;border-radius:12px;padding:1rem 1.5rem;">
        <div style="font-size:0.85rem;color:#666;">Candidate</div>
        <div style="font-size:1.2rem;font-weight:700;color:#1565C0;">{candidate_name}</div>
        <div style="font-size:0.85rem;color:#666;margin-top:0.5rem;">Role</div>
        <div style="font-weight:600;">{job_family} — {exp_level}</div>
        <div style="font-size:0.85rem;color:#666;margin-top:0.5rem;">Company</div>
        <div style="font-weight:600;">{SIZE_LABELS[company_size].split('—')[0].strip()} ({company_tier})</div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_b:
        st.markdown(f"""
        <div style="background:#E8F5E9;border-radius:12px;padding:1rem 1.5rem;">
        <div style="font-size:0.85rem;color:#666;">Predicted Salary</div>
        <div style="font-size:2rem;font-weight:800;color:#2E7D32;">${prediction:,.0f}</div>
        <div style="font-size:0.85rem;color:#666;margin-top:0.3rem;">
        Range: ${low:,.0f} – ${high:,.0f}</div>
        <div style="margin-top:0.5rem;">
        <span style="background:#2E7D32;color:white;padding:0.2rem 0.8rem;
        border-radius:20px;font-size:0.85rem;font-weight:600;">{band}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("")
 
    EXP_PROFILE = {
        "EN": {
            "title":      "Entry-Level Professional (0-2 years)",
            "summary":    "Early-career professional building foundational skills.",
            "strengths":  ["Current tooling knowledge","High learning velocity",
                           "Academic foundation","Fresh perspective on problems"],
            "gaps":       ["Limited production system experience",
                           "Stakeholder communication developing",
                           "Domain specialization still forming"],
            "trajectory": "Promotion to Mid-level within 2-3 years with consistent delivery.",
        },
        "MI": {
            "title":      "Mid-Level Professional (3-6 years)",
            "summary":    "Experienced professional capable of independent delivery.",
            "strengths":  ["Independent project ownership",
                           "Cross-functional collaboration","Proven track record"],
            "gaps":       ["Strategic thinking still developing",
                           "Limited team leadership experience"],
            "trajectory": "Senior promotion within 2-4 years depending on scope of impact.",
        },
        "SE": {
            "title":      "Senior-Level Professional (7-12 years)",
            "summary":    "Expert practitioner able to lead technical direction.",
            "strengths":  ["Deep technical expertise","Mentorship capability",
                           "System design ownership","High-impact delivery track record"],
            "gaps":       ["Organizational leadership still developing",
                           "P&L and budget ownership limited"],
            "trajectory": "Staff/Principal or management track pathway.",
        },
        "EX": {
            "title":      "Executive / Principal (13+ years)",
            "summary":    "Veteran leader driving organizational data strategy.",
            "strengths":  ["Strategic vision","Org-wide impact",
                           "Executive stakeholder management",
                           "Team building at scale"],
            "gaps":       ["Specialized compensation expectations",
                           "Limited talent pool at this level"],
            "trajectory": "C-suite, VP, Distinguished Engineer, or advisory track.",
        },
    }
    TIER_CTX = {
        "Startup":    "Lower base salary, higher equity upside. Negotiate for meaningful options.",
        "Mid-tier":   "Balanced compensation with structured bands. Bonus is limited but stable.",
        "Enterprise": "Structured salary bands with annual merit cycles. Negotiate band placement at hire.",
        "Big Tech":   "RSU refreshes dominate total compensation. Benchmark total comp, not just base.",
    }
 
    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        prof = EXP_PROFILE.get(exp_level, EXP_PROFILE["MI"])
        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm,   bottomMargin=2*cm
        )
        story = []
 
        DARK   = colors.HexColor("#0F1B5F")
        MID    = colors.HexColor("#1565C0")
        ACCENT = colors.HexColor("#1976D2")
        GREEN  = colors.HexColor("#1B5E20")
        LIGHT  = colors.HexColor("#E3F2FD")
        LGREY  = colors.HexColor("#F5F5F5")
        WHITE  = colors.white
 
        t_s = ParagraphStyle("ts", fontSize=24, textColor=WHITE,
                              alignment=TA_CENTER, fontName="Helvetica-Bold")
        s_s = ParagraphStyle("ss", fontSize=10, textColor=LIGHT,
                              alignment=TA_CENTER, fontName="Helvetica")
        h2  = ParagraphStyle("h2", fontSize=13, textColor=DARK,
                              fontName="Helvetica-Bold",
                              spaceBefore=12, spaceAfter=6)
        h3  = ParagraphStyle("h3", fontSize=11, textColor=MID,
                              fontName="Helvetica-Bold",
                              spaceBefore=8, spaceAfter=4)
        bs  = ParagraphStyle("bs", fontSize=10, textColor=colors.black,
                              fontName="Helvetica", spaceAfter=3, leading=14)
        sal = ParagraphStyle("sal", fontSize=30, textColor=WHITE,
                              alignment=TA_CENTER, fontName="Helvetica-Bold")
        bnd = ParagraphStyle("bnd", fontSize=12, textColor=WHITE,
                              alignment=TA_CENTER, fontName="Helvetica")
        sms = ParagraphStyle("sms", fontSize=8,
                              textColor=colors.HexColor("#757575"),
                              fontName="Helvetica")
 
        def sec(text):
            story.append(Spacer(1, 0.25*cm))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                    color=MID, spaceAfter=4))
            story.append(Paragraph(text, h2))
 
        # Header
        ht = Table([[Paragraph("SALARY ELIGIBILITY REPORT", t_s)]],
                   colWidths=[17*cm])
        ht.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), DARK),
            ("TOPPADDING",    (0,0),(-1,-1), 16),
            ("BOTTOMPADDING", (0,0),(-1,-1), 16),
        ]))
        story.append(ht)
        story.append(Spacer(1, 0.15*cm))
 
        sub = Table([[
            Paragraph(f"Candidate: {candidate_name}", s_s),
            Paragraph(
                f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
                s_s),
        ]], colWidths=[8.5*cm, 8.5*cm])
        sub.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), MID),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ]))
        story.append(sub)
        story.append(Spacer(1, 0.35*cm))
 
        # Salary box
        sbt = Table([
            [Paragraph(f"${prediction:,.0f}", sal)],
            [Paragraph(f"Estimated Annual Base Salary  |  {band}", bnd)],
            [Paragraph(
                f"Range: ${low:,.0f} - ${high:,.0f}  "
                f"(+/- MAE ${mae:,.0f})", bnd)],
        ], colWidths=[17*cm])
        sbt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), GREEN),
            ("TOPPADDING",    (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ]))
        story.append(sbt)
        story.append(Spacer(1, 0.2*cm))
 
        bkt = Table([[
            "Monthly",  f"${monthly:,.0f}",
            "Weekly",   f"${weekly:,.0f}",
            "Hourly",   f"${hourly:,.0f}",
        ]], colWidths=[2.5*cm, 3*cm, 2.5*cm, 3*cm, 2.5*cm, 3*cm])
        bkt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), LIGHT),
            ("FONTNAME",      (0,0),(-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 10),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ]))
        story.append(bkt)
 
        # Candidate Profile
        sec("1. CANDIDATE PROFILE")
        pd_data = [
            ["Experience", prof["title"],        "Years",      f"{years_exp} yrs"],
            ["Job Family", job_family,            "Education",  education],
            ["Company",    company_tier,          "Industry",   industry],
            ["Country",    COUNTRY_NAMES.get(country, country),
             "City Tier",  city_tier],
            ["Remote",     f"{remote_pct}%",      "Year",       str(work_year)],
        ]
        pdt = Table(pd_data, colWidths=[3.2*cm, 5.3*cm, 3.2*cm, 5.3*cm])
        pdt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,-1), LIGHT),
            ("BACKGROUND",    (2,0),(2,-1), LIGHT),
            ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (2,0),(2,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, LGREY]),
        ]))
        story.append(pdt)
 
        # Salary Breakdown
        sec("2. SALARY COMPONENT BREAKDOWN")
        ml_a = prediction - signal
        cd = [
            ["Component",     "Factor",          "Impact",            "Description"],
            ["Base (Role)",   "-",               f"${base:,.0f}",     f"{job_family} US benchmark"],
            ["Experience",    f"x{exp_mult:.3f}",
             f"+${base*(exp_mult-1):,.0f}",      f"{years_exp} years non-linear curve"],
            ["Company",       f"x{comp_m:.2f}",
             f"${base*exp_mult*(comp_m-1):+,.0f}", company_tier],
            ["Geography",     f"x{geo_m:.2f}",
             f"${base*exp_mult*comp_m*(geo_m-1):+,.0f}", country],
            ["Education",     f"x{edu_m:.2f}",
             f"+${base*exp_mult*comp_m*geo_m*(edu_m-1):,.0f}", education],
            ["Industry",      f"x{ind_m:.2f}",
             f"${base*exp_mult*comp_m*geo_m*edu_m*(ind_m-1):+,.0f}", industry],
            ["Economic Signal", "-",             f"${signal:,.0f}",   "Formula estimate"],
            ["ML Correction", "-",               f"${ml_a:+,.0f}",   "Model adjustment"],
            ["FINAL",         "-",               f"${prediction:,.0f}", band],
        ]
        cdt = Table(cd, colWidths=[3.8*cm, 2.2*cm, 2.8*cm, 8.2*cm])
        cdt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  DARK),
            ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
            ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-2), [WHITE, LGREY]),
            ("BACKGROUND",    (0,-1),(-1,-1), colors.HexColor("#E8F5E9")),
            ("FONTNAME",      (0,-1),(-1,-1), "Helvetica-Bold"),
        ]))
        story.append(cdt)
 
        # Eligibility Assessment
        sec("3. ELIGIBILITY ASSESSMENT")
        story.append(Paragraph(f"<b>{prof['title']}</b>", h3))
        story.append(Paragraph(prof["summary"], bs))
        story.append(Paragraph("<b>Key Strengths:</b>", h3))
        for s in prof["strengths"]:
            story.append(Paragraph(f"  + {s}", bs))
        story.append(Paragraph("<b>Development Areas:</b>", h3))
        for g in prof["gaps"]:
            story.append(Paragraph(f"  > {g}", bs))
        story.append(Paragraph("<b>Career Trajectory:</b>", h3))
        story.append(Paragraph(prof["trajectory"], bs))
 
        # Company Context
        sec("4. COMPANY CONTEXT")
        story.append(Paragraph(
            f"<b>{company_tier}:</b> {TIER_CTX.get(company_tier, '')}", bs))
 
        # Negotiation
        sec("5. NEGOTIATION GUIDANCE")
        ngt = Table([
            ["Strategy",      "Target Salary",           "When to Use"],
            ["Conservative",  f"${prediction*0.90:,.0f}", "First offer, limited leverage"],
            ["Market Rate",   f"${prediction:,.0f}",      "Standard negotiation"],
            ["Stretch Ask",   f"${prediction*1.12:,.0f}", "Competing offers, proven impact"],
        ], colWidths=[4*cm, 4*cm, 9*cm])
        ngt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  ACCENT),
            ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGREY]),
        ]))
        story.append(ngt)
 
        # Model Info
        sec("6. MODEL INFORMATION")
        mi_data = [
            ["Model",    meta["best_model"],     "R²",    f"{meta['r2']:.4f}"],
            ["CV R²",    f"{meta['cv_r2']:.4f}", "MAE",   f"${meta['mae']:,.0f}"],
            ["MAPE",     f"{meta['mape']:.1f}%", "Trained on", f"{meta['n_train']:,} records"],
        ]
        mit = Table(mi_data, colWidths=[3.2*cm, 5.3*cm, 3.2*cm, 5.3*cm])
        mit.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,-1), LIGHT),
            ("BACKGROUND",    (2,0),(2,-1), LIGHT),
            ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (2,0),(2,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, LGREY]),
        ]))
        story.append(mit)
 
        # Disclaimer
        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#BDBDBD")))
        story.append(Spacer(1, 0.15*cm))
        story.append(Paragraph(
            f"DISCLAIMER: This report is generated by CompensIQ, an ML model trained on "
            f"49,495 salary records. Predictions carry +/-${mae:,.0f} average error. "
            "Actual salaries vary based on individual negotiation, exact company, "
            "stock compensation, and prevailing market conditions. "
            "Use as a reference guide, not a guarantee. "
            "GitHub: github.com/Jaswanth-ak/compensiq",
            sms
        ))
 
        doc.build(story)
        buf.seek(0)
 
        fname = f"compensiq_report_{candidate_name.replace(' ','_').lower()}.pdf"
        st.download_button(
            label       = "Download PDF Report",
            data        = buf,
            file_name   = fname,
            mime        = "application/pdf",
            use_container_width=True
        )
        st.success(f"Report ready — {fname}")
 
# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    """<div style="text-align:center;color:#888;font-size:0.82rem;padding:0.5rem 0;">
    CompensIQ &nbsp;|&nbsp; Built by Jaswanth B &nbsp;|&nbsp;
    <a href="https://github.com/Jaswanth-ak/compensiq" style="color:#1565C0;">
    GitHub</a> &nbsp;|&nbsp;
    Model: HistGradientBoosting &nbsp;|&nbsp; R² 0.8948 &nbsp;|&nbsp; MAE $15,061
    </div>""",
    unsafe_allow_html=True
)