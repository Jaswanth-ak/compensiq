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
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.markdown("""
<style>
.salary-box {
    background: linear-gradient(135deg, #1A237E 0%, #1565C0 100%);
    padding: 2rem; border-radius: 1rem;
    text-align: center; color: white; margin: 1rem 0;
}
.salary-box h1 { font-size: 3.5rem; margin: 0; font-weight: 800; }
.salary-box p  { font-size: 1.1rem; opacity: 0.9; margin: 0.3rem 0; }
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #1565C0;
    padding: 6px; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 0.5rem 1.2rem;
    font-weight: 600; color: white !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1565C0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

st.title("CompensIQ - Data Science Salary Intelligence System")
st.markdown(
    f"**{meta['best_model']}** - "
    f"R2 = **{meta['r2']:.3f}** - "
    f"MAE = **${meta['mae']:,.0f}** - "
    f"MAPE = **{meta['mape']:.1f}%** - "
    f"Trained on **{meta['n_train']:,}** samples"
)
st.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Jaswanth--ak%2Fcompensiq-black?logo=github)]"
    "(https://github.com/Jaswanth-ak/compensiq)",
    unsafe_allow_html=True
)
st.divider()

REGION_MAP = {
    "US":"North America","CA":"North America","MX":"North America",
    "GB":"Europe_West","DE":"Europe_West","FR":"Europe_West",
    "NL":"Europe_West","IE":"Europe_West","AT":"Europe_West",
    "BE":"Europe_West","SE":"Europe_West","NO":"Europe_West",
    "DK":"Europe_West","FI":"Europe_West","CH":"Europe_West",
    "ES":"Europe_South","IT":"Europe_South","PT":"Europe_South",
    "GR":"Europe_South",
    "PL":"Europe_East","CZ":"Europe_East","RO":"Europe_East",
    "HU":"Europe_East","SK":"Europe_East","LT":"Europe_East",
    "LV":"Europe_East","EE":"Europe_East","HR":"Europe_East",
    "BG":"Europe_East","RS":"Europe_East","UA":"Europe_East",
    "IN":"Asia_Em","CN":"Asia_Em","PK":"Asia_Em","BD":"Asia_Em",
    "PH":"Asia_Em","VN":"Asia_Em","MY":"Asia_Em",
    "ID":"Asia_Em","TH":"Asia_Em",
    "JP":"Asia_Dev","KR":"Asia_Dev","SG":"Asia_Dev",
    "AU":"Asia_Dev","NZ":"Asia_Dev","HK":"Asia_Dev","TW":"Asia_Dev",
    "IL":"Middle_East","AE":"Middle_East","SA":"Middle_East",
    "TR":"Middle_East","QA":"Middle_East",
    "BR":"LatAm","AR":"LatAm","CO":"LatAm","CL":"LatAm","PE":"LatAm",
    "NG":"Africa","KE":"Africa","ZA":"Africa","GH":"Africa","EG":"Africa",
}

COUNTRY_GEO = {
    "US":1.000,"CA":0.780,"MX":0.290,
    "GB":0.700,"DE":0.680,"FR":0.600,"NL":0.730,
    "IE":0.750,"AT":0.640,"BE":0.620,"SE":0.650,
    "NO":0.700,"DK":0.680,"FI":0.610,"CH":0.920,
    "ES":0.460,"IT":0.450,"PT":0.380,"GR":0.340,
    "PL":0.320,"CZ":0.340,"RO":0.250,"HU":0.280,
    "SK":0.280,"LT":0.240,"LV":0.240,"EE":0.290,
    "HR":0.260,"BG":0.210,"RS":0.220,"UA":0.160,
    "IN":0.180,"CN":0.360,"PK":0.140,"BD":0.120,
    "PH":0.200,"VN":0.150,"MY":0.260,"ID":0.180,"TH":0.230,
    "JP":0.520,"KR":0.480,"SG":0.680,"AU":0.740,
    "NZ":0.610,"HK":0.620,"TW":0.420,
    "IL":0.620,"AE":0.580,"SA":0.490,"TR":0.220,"QA":0.510,
    "BR":0.250,"AR":0.180,"CO":0.210,"CL":0.290,"PE":0.200,
    "NG":0.140,"KE":0.150,"ZA":0.250,"GH":0.130,"EG":0.160,
}

COUNTRY_NAMES = {
    "US":"United States","CA":"Canada","MX":"Mexico",
    "GB":"United Kingdom","DE":"Germany","FR":"France",
    "NL":"Netherlands","IE":"Ireland","AT":"Austria",
    "BE":"Belgium","SE":"Sweden","NO":"Norway",
    "DK":"Denmark","FI":"Finland","CH":"Switzerland",
    "ES":"Spain","IT":"Italy","PT":"Portugal","GR":"Greece",
    "PL":"Poland","CZ":"Czech Republic","RO":"Romania",
    "HU":"Hungary","SK":"Slovakia","LT":"Lithuania",
    "LV":"Latvia","EE":"Estonia","HR":"Croatia",
    "BG":"Bulgaria","RS":"Serbia","UA":"Ukraine",
    "IN":"India","CN":"China","PK":"Pakistan",
    "BD":"Bangladesh","PH":"Philippines","VN":"Vietnam",
    "MY":"Malaysia","ID":"Indonesia","TH":"Thailand",
    "JP":"Japan","KR":"South Korea","SG":"Singapore",
    "AU":"Australia","NZ":"New Zealand","HK":"Hong Kong","TW":"Taiwan",
    "IL":"Israel","AE":"UAE","SA":"Saudi Arabia",
    "TR":"Turkey","QA":"Qatar",
    "BR":"Brazil","AR":"Argentina","CO":"Colombia",
    "CL":"Chile","PE":"Peru",
    "NG":"Nigeria","KE":"Kenya","ZA":"South Africa",
    "GH":"Ghana","EG":"Egypt",
}

REGION_FALLBACK = {
    "North America":0.820,"Europe_West":0.660,"Europe_South":0.420,
    "Europe_East":0.270,"Asia_Dev":0.570,"Asia_Em":0.220,
    "Middle_East":0.490,"LatAm":0.230,"Africa":0.170,"Other":0.300,
}

BASE_SAL = {
    "ML Engineer":128000,"Data Scientist":118000,
    "Manager":140000,"Architect":135000,
    "Data Engineer":115000,"Analyst":88000,"Other":100000,
}

def get_tier(size, job_family):
    if size == "S": return "Startup"
    if size == "M": return "Mid-tier"
    if job_family in ["ML Engineer","Data Scientist"]: return "Big Tech"
    return "Enterprise"

TIER_MULT = {"Startup":0.82,"Mid-tier":1.00,"Enterprise":1.12,"Big Tech":1.45}
EDU_M     = {"Bachelor":1.00,"Master":1.08,"PhD":1.18,"MBA":1.10}
IND_M     = {"Tech":1.00,"Finance":1.22,"Research":1.05,
              "Consulting":1.08,"Other_Industry":0.90}
MARKET    = {2020:0.91,2021:0.94,2022:1.06,
              2023:1.09,2024:1.00,2025:0.98,2026:1.01}
EXP_BOUNDS = {"EN":(0,2,1),"MI":(3,6,4),"SE":(7,12,9),"EX":(13,22,16)}
SIZE_LABELS = {
    "S":"Small  (Startup, <50 employees)",
    "M":"Medium (Mid-tier, 50-999 employees)",
    "L":"Large  (Enterprise / Big Tech, 1000+)",
}

def exp_multiplier(years):
    m = 1.0
    for yr in range(1, int(years)+1):
        if yr<=2: m*=1.060
        elif yr<=5: m*=1.045
        elif yr<=8: m*=1.030
        elif yr<=12: m*=1.018
        else: m*=1.010
    return m

st.sidebar.header("Candidate Profile")
candidate_name = st.sidebar.text_input("Candidate Name", "Your Name")

st.sidebar.subheader("Experience")
exp_level = st.sidebar.selectbox(
    "Experience Level",
    ["EN","MI","SE","EX"],
    format_func=lambda x: {
        "EN":"Entry (0-2yr)","MI":"Mid (3-6yr)",
        "SE":"Senior (7-12yr)","EX":"Executive (13+yr)"
    }[x], index=2
)
exp_lo, exp_hi, exp_def = EXP_BOUNDS[exp_level]
years_exp = st.sidebar.slider(
    "Years of Experience",
    min_value=exp_lo, max_value=exp_hi,
    value=min(exp_def, exp_hi)
)
st.sidebar.caption(f"Valid range for {exp_level}: {exp_lo}-{exp_hi} years")

st.sidebar.subheader("Role")
job_family = st.sidebar.selectbox(
    "Job Family",
    ["Data Scientist","ML Engineer","Data Engineer",
     "Analyst","Manager","Architect","Other"]
)
education = st.sidebar.selectbox(
    "Education", ["Bachelor","Master","PhD","MBA"]
)

st.sidebar.subheader("Company")
company_size = st.sidebar.selectbox(
    "Company Size",
    ["S","M","L"],
    format_func=lambda x: SIZE_LABELS[x],
    index=1
)
company_tier = get_tier(company_size, job_family)
st.sidebar.caption(f"Tier: {company_tier}  |  Multiplier: x{TIER_MULT[company_tier]:.2f}")

st.sidebar.subheader("Location")
country = st.sidebar.selectbox(
    "Country",
    list(COUNTRY_NAMES.keys()),
    format_func=lambda x: f"{COUNTRY_NAMES[x]} ({x})"
)

if country == "US":
    city_tier = "Tier2_City"
else:
    city_tier = "International"

st.sidebar.subheader("Work Setup")
industry = st.sidebar.selectbox(
    "Industry",
    ["Tech","Finance","Research","Consulting","Other_Industry"],
    format_func=lambda x: {
        "Tech":"Tech",
        "Finance":"Finance  (+22% premium)",
        "Research":"Research  (+5% premium)",
        "Consulting":"Consulting  (+8% premium)",
        "Other_Industry":"Other Industry  (-10%)"
    }[x]
)
work_year = st.sidebar.selectbox("Year", [2023,2024,2025,2026], index=2)

region   = REGION_MAP.get(country, "Other")
geo_m    = COUNTRY_GEO.get(country, REGION_FALLBACK.get(region, 0.300))
is_us    = 1 if country == "US" else 0
base     = BASE_SAL.get(job_family, 100000)
comp_m   = TIER_MULT.get(company_tier, 1.0)
edu_m    = EDU_M.get(education, 1.0)
ind_m    = IND_M.get(industry, 1.0)
rem_m    = 1.00
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
    "employee_region":region,"remote_type":"onsite",
    "years_of_experience":years_exp,"seniority_flag":snr,
    "is_us":is_us,"geo_multiplier":geo_m,
    "remote_fraction":0.0,"years_since_2020":yrs_2020,
    "market_cycle":mkt_cyc,"is_2026":is_2026f,
    "base_salary_index":base,"company_multiplier":comp_m,
    "edu_multiplier":edu_m,"industry_multiplier":ind_m,
    "remote_multiplier":rem_m,"experience_multiplier":exp_mult,
    "skills_multiplier":skill_m,"expected_salary_signal":signal,
    "log_expected_signal":log_sig,"exp_x_geo":years_exp*geo_m,
    "exp_x_company":years_exp*comp_m,"exp_x_edu":years_exp*edu_m,
    "seniority_x_exp":snr*years_exp,"geo_x_tier":geo_m*comp_m,
    "signal_x_market":signal*mkt_cyc,
}

idf  = pd.DataFrame([row])
ienc = pd.get_dummies(idf, columns=cat_cols, drop_first=True)
ienc = ienc.reindex(columns=feature_cols, fill_value=0)
pred = float(np.expm1(model.predict(ienc)[0]))
prediction = max(pred, 15000)

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

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction", "Breakdown", "Model Info", "Download Report"
])

with tab1:
    st.markdown(f"""
    <div class="salary-box">
        <p>Estimated Annual Base Salary</p>
        <h1>${prediction:,.0f}</h1>
        <p>{band}</p>
        <p>Range: ${low:,.0f} - ${high:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Monthly", f"${monthly:,.0f}")
    c2.metric("Weekly",  f"${weekly:,.0f}")
    c3.metric("Hourly",  f"${hourly:,.0f}")
    c4.metric("MAE +-",  f"${mae:,.0f}")

    st.subheader("Salary Ladders")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**By Experience Level (same company and role)**")
        for lvl, yrs, label in [
            ("EN",1,"Entry"),("MI",4,"Mid"),
            ("SE",9,"Senior"),("EX",16,"Exec")
        ]:
            snr_  = 1 if yrs>=7 else 0
            em_   = exp_multiplier(yrs)
            sm_   = 1.0+yrs*0.006
            sig_  = base*em_*comp_m*geo_m*edu_m*ind_m*rem_m*yr_m
            r_ = {
                "experience_level":lvl,"job_family":job_family,
                "company_size":company_size,"company_tier":company_tier,
                "education_level":education,"industry":industry,
                "city_tier":city_tier,"company_region":region,
                "employee_region":region,"remote_type":"onsite",
                "years_of_experience":yrs,"seniority_flag":snr_,
                "is_us":is_us,"geo_multiplier":geo_m,
                "remote_fraction":0.0,
                "years_since_2020":yrs_2020,"market_cycle":mkt_cyc,
                "is_2026":is_2026f,"base_salary_index":base,
                "company_multiplier":comp_m,"edu_multiplier":edu_m,
                "industry_multiplier":ind_m,"remote_multiplier":rem_m,
                "experience_multiplier":em_,"skills_multiplier":sm_,
                "expected_salary_signal":sig_,
                "log_expected_signal":np.log1p(sig_),
                "exp_x_geo":yrs*geo_m,"exp_x_company":yrs*comp_m,
                "exp_x_edu":yrs*edu_m,"seniority_x_exp":snr_*yrs,
                "geo_x_tier":geo_m*comp_m,"signal_x_market":sig_*mkt_cyc,
            }
            df_ = pd.DataFrame([r_])
            e_  = pd.get_dummies(df_, columns=cat_cols, drop_first=True)
            e_  = e_.reindex(columns=feature_cols, fill_value=0)
            p_  = max(float(np.expm1(model.predict(e_)[0])),15000)
            marker = "  << YOU" if lvl==exp_level else ""
            st.markdown(f"`{label:<8}` ${p_:>9,.0f}{marker}")

    with col2:
        st.markdown("**By Company Size (same role and experience)**")
        for sz_,label_ in [
            ("S","Small  (Startup)"),
            ("M","Medium (Mid-tier)"),
            ("L","Large  (Enterprise/BigTech)"),
        ]:
            tr_  = get_tier(sz_, job_family)
            cm_  = TIER_MULT[tr_]
            sig_ = base*exp_mult*cm_*geo_m*edu_m*ind_m*rem_m*yr_m
            r_ = {
                "experience_level":exp_level,"job_family":job_family,
                "company_size":sz_,"company_tier":tr_,
                "education_level":education,"industry":industry,
                "city_tier":city_tier,"company_region":region,
                "employee_region":region,"remote_type":"onsite",
                "years_of_experience":years_exp,"seniority_flag":snr,
                "is_us":is_us,"geo_multiplier":geo_m,
                "remote_fraction":0.0,
                "years_since_2020":yrs_2020,"market_cycle":mkt_cyc,
                "is_2026":is_2026f,"base_salary_index":base,
                "company_multiplier":cm_,"edu_multiplier":edu_m,
                "industry_multiplier":ind_m,"remote_multiplier":rem_m,
                "experience_multiplier":exp_mult,"skills_multiplier":skill_m,
                "expected_salary_signal":sig_,
                "log_expected_signal":np.log1p(sig_),
                "exp_x_geo":years_exp*geo_m,"exp_x_company":years_exp*cm_,
                "exp_x_edu":years_exp*edu_m,"seniority_x_exp":snr*years_exp,
                "geo_x_tier":geo_m*cm_,"signal_x_market":sig_*mkt_cyc,
            }
            df_ = pd.DataFrame([r_])
            e_  = pd.get_dummies(df_, columns=cat_cols, drop_first=True)
            e_  = e_.reindex(columns=feature_cols, fill_value=0)
            p_  = max(float(np.expm1(model.predict(e_)[0])),15000)
            marker = "  << YOU" if sz_==company_size else ""
            st.markdown(f"`{label_:<25}` ${p_:>9,.0f}{marker}")

with tab2:
    st.subheader("Salary Component Breakdown")
    exp_impact  = base*(exp_mult-1)
    comp_impact = base*exp_mult*(comp_m-1)
    geo_impact  = base*exp_mult*comp_m*(geo_m-1)
    edu_impact  = base*exp_mult*comp_m*geo_m*(edu_m-1)
    ind_impact  = base*exp_mult*comp_m*geo_m*edu_m*(ind_m-1)
    ml_adj      = prediction-signal

    components = [
        ("Base Salary",     f"${base:,.0f}",
         f"{job_family} US benchmark"),
        ("Experience",      f"+${exp_impact:,.0f}",
         f"{years_exp} years, x{exp_mult:.3f}"),
        ("Company Size",    f"${comp_impact:+,.0f}",
         f"{SIZE_LABELS[company_size]} - {company_tier}, x{comp_m:.2f}"),
        ("Geography",       f"${geo_impact:+,.0f}",
         f"{COUNTRY_NAMES.get(country,country)}, x{geo_m:.2f}"),
        ("Education",       f"+${edu_impact:,.0f}",
         f"{education}, x{edu_m:.2f}"),
        ("Industry",        f"${ind_impact:+,.0f}",
         f"{industry}, x{ind_m:.2f}"),
        ("Economic Signal", f"${signal:,.0f}",
         "Formula-based estimate"),
        ("ML Correction",   f"${ml_adj:+,.0f}",
         "Pattern learned from 49,495 salary records"),
    ]
    for label,value,desc in components:
        a,b,c = st.columns([3,2,4])
        a.write(label)
        b.write(f"**{value}**")
        c.write(desc)
        st.divider()

    st.success(f"Final Prediction: ${prediction:,.0f}  |  {band}")

    st.subheader("Negotiation Range")
    n1,n2,n3 = st.columns(3)
    n1.metric("Conservative", f"${prediction*0.90:,.0f}", "-10%")
    n2.metric("Market Rate",  f"${prediction:,.0f}",      "Your profile")
    n3.metric("Stretch Ask",  f"${prediction*1.12:,.0f}", "+12%")

with tab3:
    st.subheader("Model Performance")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("R2",   f"{meta['r2']:.4f}")
    m2.metric("CV R2",f"{meta['cv_r2']:.4f}")
    m3.metric("MAE",  f"${meta['mae']:,.0f}")
    m4.metric("MAPE", f"{meta['mape']:.1f}%")

    st.markdown("---")
    st.subheader("All Models Comparison")
    if "all_models" in meta:
        rows = []
        for name,m in meta["all_models"].items():
            rows.append({
                "Model":name,"R2":m["r2"],
                "MAE":f"${m['mae']:,.0f}",
                "MAPE":f"{m['mape']:.1f}%",
                "Best":"YES" if name==meta["best_model"] else ""
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Charts")
    for img,title in [
        ("charts/03_diagnostics.png",         "Actual vs Predicted"),
        ("charts/04_band_performance.png",    "Performance by Salary Band"),
        ("charts/05_feature_importance.png",  "Feature Importance"),
        ("charts/06_model_comparison.png",    "Model Comparison"),
        ("charts/07_salary_distributions.png","Salary Distributions"),
    ]:
        if os.path.exists(img):
            st.image(img, caption=title, use_container_width=True)

with tab4:
    st.subheader("Download Candidate Eligibility Report")
    st.markdown(
        "Generate a professional PDF with full salary breakdown, "
        "eligibility assessment, and negotiation guidance."
    )

    EXP_PROFILE = {
        "EN":{"title":"Entry-Level Professional (0-2 years)",
              "summary":"Early-career professional building foundational skills.",
              "strengths":["Current tooling knowledge","High learning velocity",
                           "Academic foundation","Fresh perspective"],
              "gaps":["Limited production experience",
                      "Stakeholder communication developing",
                      "Domain specialization forming"],
              "trajectory":"Promotion to Mid-level within 2-3 years."},
        "MI":{"title":"Mid-Level Professional (3-6 years)",
              "summary":"Experienced professional capable of independent delivery.",
              "strengths":["Independent project ownership",
                           "Cross-functional collaboration","Proven track record"],
              "gaps":["Strategic thinking developing","Limited team leadership"],
              "trajectory":"Senior promotion within 2-4 years."},
        "SE":{"title":"Senior-Level Professional (7-12 years)",
              "summary":"Expert practitioner able to lead technical direction.",
              "strengths":["Deep technical expertise","Mentorship capability",
                           "System design ownership","High-impact delivery"],
              "gaps":["Org leadership developing","P&L ownership limited"],
              "trajectory":"Staff/Principal or management track."},
        "EX":{"title":"Executive / Principal (13+ years)",
              "summary":"Veteran leader driving organizational data strategy.",
              "strengths":["Strategic vision","Org-wide impact",
                           "Executive stakeholder management"],
              "gaps":["Specialized compensation expectations",
                      "Limited talent pool"],
              "trajectory":"C-suite, VP, or advisory track."},
    }
    TIER_CTX = {
        "Startup"   :"Lower base, higher equity. Negotiate for meaningful options.",
        "Mid-tier"  :"Balanced bands. Bonus is limited but stable.",
        "Enterprise":"Structured bands. Negotiate placement at hire.",
        "Big Tech"  :"RSU refreshes dominate total comp. Benchmark total compensation.",
    }

    if st.button("Generate and Download PDF Report", type="primary"):
        prof = EXP_PROFILE.get(exp_level, EXP_PROFILE["MI"])
        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm,   bottomMargin=2*cm
        )
        story  = []
        DARK   = colors.HexColor("#1A237E")
        MID    = colors.HexColor("#283593")
        ACCENT = colors.HexColor("#1565C0")
        GREEN  = colors.HexColor("#2E7D32")
        LIGHT  = colors.HexColor("#E3F2FD")
        LGREY  = colors.HexColor("#F5F5F5")
        WHITE  = colors.white

        t_s = ParagraphStyle("ts",fontSize=22,textColor=WHITE,
                              alignment=TA_CENTER,fontName="Helvetica-Bold")
        s_s = ParagraphStyle("ss",fontSize=10,textColor=LIGHT,
                              alignment=TA_CENTER,fontName="Helvetica")
        h2  = ParagraphStyle("h2",fontSize=13,textColor=DARK,
                              fontName="Helvetica-Bold",spaceBefore=10,spaceAfter=5)
        h3  = ParagraphStyle("h3",fontSize=11,textColor=ACCENT,
                              fontName="Helvetica-Bold",spaceBefore=7,spaceAfter=3)
        bs  = ParagraphStyle("bs",fontSize=10,textColor=colors.black,
                              fontName="Helvetica",spaceAfter=3,leading=14)
        ss  = ParagraphStyle("ss2",fontSize=28,textColor=WHITE,
                              alignment=TA_CENTER,fontName="Helvetica-Bold")
        bns = ParagraphStyle("bns",fontSize=11,textColor=WHITE,
                              alignment=TA_CENTER,fontName="Helvetica")
        sms = ParagraphStyle("sms",fontSize=8,
                              textColor=colors.HexColor("#616161"),
                              fontName="Helvetica")

        def sec(text):
            story.append(Spacer(1,0.25*cm))
            story.append(HRFlowable(width="100%",thickness=1,
                                    color=ACCENT,spaceAfter=3))
            story.append(Paragraph(text,h2))

        ht = Table([[Paragraph("SALARY ELIGIBILITY REPORT",t_s)]],
                   colWidths=[17*cm])
        ht.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),DARK),
            ("TOPPADDING",(0,0),(-1,-1),14),
            ("BOTTOMPADDING",(0,0),(-1,-1),14),
        ]))
        story.append(ht)
        story.append(Spacer(1,0.15*cm))

        st2 = Table([[
            Paragraph(f"Candidate: {candidate_name}",s_s),
            Paragraph(
                f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
                s_s),
        ]],colWidths=[8.5*cm,8.5*cm])
        st2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),MID),
            ("TOPPADDING",(0,0),(-1,-1),7),
            ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ]))
        story.append(st2)
        story.append(Spacer(1,0.3*cm))

        sbt = Table([
            [Paragraph(f"${prediction:,.0f}",ss)],
            [Paragraph(f"Estimated Annual Base Salary | {band}",bns)],
            [Paragraph(
                f"Range: ${low:,.0f} - ${high:,.0f} "
                f"(+/- MAE ${mae:,.0f})",bns)],
        ],colWidths=[17*cm])
        sbt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),GREEN),
            ("TOPPADDING",(0,0),(-1,-1),8),
            ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ]))
        story.append(sbt)
        story.append(Spacer(1,0.2*cm))

        bkt = Table([[
            "Monthly",f"${monthly:,.0f}",
            "Weekly",f"${weekly:,.0f}",
            "Hourly",f"${hourly:,.0f}",
        ]],colWidths=[2.5*cm,3*cm,2.5*cm,3*cm,2.5*cm,3*cm])
        bkt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),LIGHT),
            ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),10),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),7),
            ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ]))
        story.append(bkt)

        sec("1. CANDIDATE PROFILE")
        pd_data = [
            ["Experience",prof["title"],"Years",f"{years_exp} yrs"],
            ["Job Family",job_family,"Education",education],
            ["Company",f"{SIZE_LABELS[company_size]}","Industry",industry],
            ["Country",COUNTRY_NAMES.get(country,country),
             "Geo Multiplier",f"x{geo_m:.2f}"],
            ["Year",str(work_year),"Company Tier",company_tier],
        ]
        pdt = Table(pd_data,colWidths=[3.2*cm,5.3*cm,3.2*cm,5.3*cm])
        pdt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(0,-1),LIGHT),
            ("BACKGROUND",(2,0),(2,-1),LIGHT),
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),6),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[WHITE,LGREY]),
        ]))
        story.append(pdt)

        sec("2. SALARY BREAKDOWN")
        ml_a = prediction-signal
        cd = [
            ["Component","Factor","Impact","Description"],
            ["Base (Role)","—",f"${base:,.0f}",f"{job_family} benchmark"],
            ["Experience",f"x{exp_mult:.3f}",
             f"+${base*(exp_mult-1):,.0f}",f"{years_exp} years"],
            ["Company",f"x{comp_m:.2f}",
             f"${base*exp_mult*(comp_m-1):+,.0f}",company_tier],
            ["Geography",f"x{geo_m:.2f}",
             f"${base*exp_mult*comp_m*(geo_m-1):+,.0f}",country],
            ["Education",f"x{edu_m:.2f}",
             f"+${base*exp_mult*comp_m*geo_m*(edu_m-1):,.0f}",education],
            ["Industry",f"x{ind_m:.2f}",
             f"${base*exp_mult*comp_m*geo_m*edu_m*(ind_m-1):+,.0f}",industry],
            ["Signal","—",f"${signal:,.0f}","Formula estimate"],
            ["ML Adj","—",f"${ml_a:+,.0f}","Model correction"],
            ["FINAL","—",f"${prediction:,.0f}",band],
        ]
        cdt = Table(cd,colWidths=[3.8*cm,2.2*cm,2.8*cm,8.2*cm])
        cdt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),DARK),
            ("TEXTCOLOR",(0,0),(-1,0),WHITE),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),5),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-2),[WHITE,LGREY]),
            ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#E8F5E9")),
            ("FONTNAME",(0,-1),(-1,-1),"Helvetica-Bold"),
        ]))
        story.append(cdt)

        sec("3. ELIGIBILITY ASSESSMENT")
        story.append(Paragraph(f"<b>{prof['title']}</b>",h3))
        story.append(Paragraph(prof["summary"],bs))
        story.append(Paragraph("<b>Strengths:</b>",h3))
        for s in prof["strengths"]:
            story.append(Paragraph(f"  + {s}",bs))
        story.append(Paragraph("<b>Development Areas:</b>",h3))
        for g in prof["gaps"]:
            story.append(Paragraph(f"  > {g}",bs))
        story.append(Paragraph("<b>Trajectory:</b>",h3))
        story.append(Paragraph(prof["trajectory"],bs))

        sec("4. COMPANY CONTEXT")
        story.append(Paragraph(
            f"<b>{company_tier}:</b> {TIER_CTX.get(company_tier,'')}",bs))

        sec("5. NEGOTIATION GUIDANCE")
        ngt = Table([
            ["Strategy","Target","When to Use"],
            ["Conservative",f"${prediction*0.90:,.0f}",
             "First offer, limited leverage"],
            ["Market Rate",f"${prediction:,.0f}",
             "Standard negotiation"],
            ["Stretch",f"${prediction*1.12:,.0f}",
             "Competing offers, proven impact"],
        ],colWidths=[4*cm,4*cm,9*cm])
        ngt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),ACCENT),
            ("TEXTCOLOR",(0,0),(-1,0),WHITE),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),7),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGREY]),
        ]))
        story.append(ngt)

        story.append(Spacer(1,0.3*cm))
        story.append(HRFlowable(width="100%",thickness=0.5,
                                color=colors.HexColor("#BDBDBD")))
        story.append(Spacer(1,0.15*cm))
        story.append(Paragraph(
            f"DISCLAIMER: CompensIQ ML model trained on 49,495 records. "
            f"Predictions carry +/-${mae:,.0f} average error. "
            "Actual salaries vary by negotiation, company, and market. "
            "GitHub: github.com/Jaswanth-ak/compensiq",
            sms))

        doc.build(story)
        buf.seek(0)

        fname = f"compensiq_report_{candidate_name.replace(' ','_').lower()}.pdf"
        st.download_button(
            label="Download PDF",
            data=buf,
            file_name=fname,
            mime="application/pdf"
        )
        st.success(f"Report ready - click Download PDF above")

st.divider()
st.markdown(
    """<div style="text-align:center;color:#888;font-size:0.82rem;">
    CompensIQ &nbsp;|&nbsp; Built by Jaswanth B &nbsp;|&nbsp;
    <a href="https://github.com/Jaswanth-ak/compensiq"
    style="color:#1565C0;">GitHub</a> &nbsp;|&nbsp;
    HistGradientBoosting &nbsp;|&nbsp; R2 0.8948 &nbsp;|&nbsp; MAE $15,061
    </div>""",
    unsafe_allow_html=True
)