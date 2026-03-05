import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. HELPER FUNCTIONS (Must match App logic)
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
# 2. LOAD DATA
# ──────────────────────────────────────────────
print("🚀 Loading Data...")
# Try reading standard CSV, if not found try default Kaggle name
try:
    df = pd.read_csv("employees.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("ds_salaries.csv")
    except:
        print("❌ Error: 'employees.csv' or 'ds_salaries.csv' not found.")
        exit()

print(f"   Data Shape: {df.shape}")

# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("🛠️  Engineering Features...")

# 3.1 Time based
df['years_since_2020'] = df['work_year'] - 2020

# 3.2 Maps
exp_tier_map = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
df['experience_tier'] = df['experience_level'].map(exp_tier_map).fillna("Mid")

company_scale_map = {'S': 1, 'M': 2, 'L': 3}
df['company_scale_score'] = df['company_size'].map(company_scale_map)

# 3.3 Derived Remote features
df['remote_fraction'] = df['remote_ratio'] / 100
df['is_fully_remote'] = (df['remote_ratio'] == 100).astype(int)

# 3.4 Text Analysis Functions
df['seniority_score'] = df['job_title'].apply(seniority_score)
df['market_demand'] = df['job_title'].apply(market_demand)

# 3.5 Generate Cost of Living Index Map (Based on average salary per country in data)
# This creates the 'col_index_map' expected by the app
country_avg = df.groupby('company_location')['salary_in_usd'].mean()
# Normalize to 0-100 scale for "index" feel
col_index_map = ((country_avg / country_avg.max()) * 100).to_dict()
df['cost_of_living_index'] = df['company_location'].map(col_index_map)

# ──────────────────────────────────────────────
# 4. PREPARE TARGETS
# ──────────────────────────────────────────────
print("🎯 Preparing Targets...")

# Regression Target (Log Transformed)
df['salary_log'] = np.log1p(df['salary_in_usd'])

# Classification Target (4 Categories: Low, Medium, High, Very High)
# Using qcut to create equal-sized bins based on actual data
df['salary_band'] = pd.qcut(df['salary_in_usd'], q=4, labels=["Low", "Medium", "High", "Very High"])

# ──────────────────────────────────────────────
# 5. ENCODING
# ──────────────────────────────────────────────
print("🔢 Encoding Categories...")

encoders = {}
cat_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 
            'company_location', 'company_size', 'experience_tier']

for col in cat_cols:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col])
    encoders[col] = le

# Target Encoder for Band
target_encoder = LabelEncoder()
df['salary_band_enc'] = target_encoder.fit_transform(df['salary_band'])

# ──────────────────────────────────────────────
# 6. FEATURE SELECTION & SPLIT
# ──────────────────────────────────────────────
# STRICT ORDER MATCHING APP.PY INPUT ARRAY
feature_cols = [
    'years_since_2020',
    'experience_level_enc',
    'employment_type_enc',
    'job_title_enc',
    'employee_residence_enc',
    'company_location_enc',
    'company_size_enc',
    'experience_tier_enc',
    'remote_ratio',
    'remote_fraction',
    'is_fully_remote',
    'cost_of_living_index',
    'seniority_score',
    'market_demand',
    'company_scale_score'
]

X = df[feature_cols]
y_cls = df['salary_band_enc']
y_reg = df['salary_log']

X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)
_, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 7. TRAINING CLASSIFIERS
# ──────────────────────────────────────────────
print("\n🤖 Training Classifiers...")

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Clf": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting Clf": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_clf_score = 0
best_clf_model = None
best_clf_name = ""
clf_results = {}

for name, model in clf_models.items():
    model.fit(X_train_scaled, y_train_c)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds, average='weighted')
    
    print(f"   {name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    clf_results[name] = {"Accuracy": f"{acc:.2%}", "F1 Score": f"{f1:.2f}"}
    
    if acc > best_clf_score:
        best_clf_score = acc
        best_clf_model = model
        best_clf_name = name

# ──────────────────────────────────────────────
# 8. TRAINING REGRESSORS
# ──────────────────────────────────────────────
print("\n📉 Training Regressors...")

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Reg": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Reg": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_reg_score = -999
best_reg_model = None
best_reg_name = ""
reg_results = {}

for name, model in reg_models.items():
    model.fit(X_train_scaled, y_train_r)
    preds_log = model.predict(X_test_scaled)
    
    # Calculate metrics on LOG scale for R2, but MAE on actual dollars
    r2 = r2_score(y_test_r, preds_log)
    
    # Convert back to dollars for MAE
    preds_dollar = np.expm1(preds_log)
    actual_dollar = np.expm1(y_test_r)
    mae = mean_absolute_error(actual_dollar, preds_dollar)
    
    print(f"   {name} -> R2: {r2:.4f}, MAE: ${mae:,.0f}")
    
    reg_results[name] = {"R2 Score": f"{r2:.2f}", "MAE": f"${mae:,.0f}"}
    
    if r2 > best_reg_score:
        best_reg_score = r2
        best_reg_model = model
        best_reg_name = name

# ──────────────────────────────────────────────
# 9. SAVE PACKAGE
# ──────────────────────────────────────────────
print(f"\n📦 Saving Model Package...")
print(f"   Best Classifier: {best_clf_name}")
print(f"   Best Regressor: {best_reg_name}")

package = {
    "classifier": best_clf_model,
    "classifier_name": best_clf_name,
    "regressor": best_reg_model,
    "regressor_name": best_reg_name,
    "scaler": scaler,
    "encoders": encoders,
    "target_encoder": target_encoder,
    "col_index_map": col_index_map,
    "company_scale_map": company_scale_map,
    "exp_tier_map": exp_tier_map,
    "clf_results": clf_results,
    "reg_results": reg_results
}

joblib.dump(package, "salary_model.pkl")
print("✅ Done! 'salary_model.pkl' created successfully.")
print("   Run 'streamlit run app.py' to launch the interface.")
