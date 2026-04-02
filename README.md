# CompensIQ

**ML-powered salary intelligence for smarter hiring decisions.**

> Predict. Explain. Decide. — Eliminating manual compensation benchmarking with production-grade machine learning.

🔗 [Live Demo](https://compensiq-rzhdftchaheptc3tn22dsp.streamlit.app)

---

## Overview

CompensIQ is a production-grade salary prediction system that helps HR teams and hiring managers make data-driven, defensible compensation decisions — instantly.

It predicts salaries across **62 countries**, **7 job families**, and **7 job titles**, trained on **50,000+ real-world records**. Every prediction is fully explainable through an economic signal layer built on top of the model, and results are delivered via an interactive Streamlit web app with auto-generated PDF candidate reports.

---

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.89 |
| MAE | $15,061 |
| MAPE | 12.2% |
| Economic Signal Correlation | Pearson r = 0.93 |
| Training Records | 50,000+ |
| Countries Covered | 62 |
| Job Families | 7 |
| Job Titles | 7 |

---

## Features

- **Salary Prediction Engine** — Predicts market compensation with high accuracy across global roles and industries
- **Economic Signal Layer** — Geo multipliers, company tier mapping, and experience curves make predictions explainable and business-ready
- **Explainability First** — Every output is traceable to specific economic factors, increasing stakeholder trust and adoption
- **Auto-Generated PDF Reports** — Candidate compensation reports generated instantly, cutting turnaround from hours to seconds
- **Global Coverage** — Handles compensation variance across 62 countries with region-specific economic adjustments
- **Interactive Web App** — Clean Streamlit interface designed for HR teams with no ML expertise required

---

## How It Works

```
User Input
(job title, country, experience level, company tier)
        │
        ▼
Economic Signal Layer
├── Geo Multipliers       → adjusts for country-level cost of living & market rates
├── Company Tier Mapping  → weights compensation by company size and prestige
└── Experience Curves     → non-linear adjustment for seniority levels
        │
        ▼
Feature Engineering
(PCA, encoding, scaling)
        │
        ▼
HistGradientBoosting Model
(R² = 0.89, hyperparameter tuned, cross-validated)
        │
        ▼
Salary Prediction + Confidence Range
        │
        ▼
Auto-Generated PDF Candidate Report
```

---

## Model Details & Training Approach

### Algorithm
**HistGradientBoosting Regressor** (Scikit-learn) — chosen for its ability to handle mixed feature types, missing values natively, and large datasets efficiently while maintaining interpretability.

### Feature Engineering
- Categorical encoding for job title, country, and job family
- Custom economic signal layer (Pearson r = 0.93 with target variable)
- PCA applied for dimensionality reduction on high-cardinality features
- MinMax scaling for numerical stability

### Training & Evaluation
- Hyperparameter tuning via grid search
- Cross-validated model evaluation (k-fold) to prevent overfitting
- Evaluated on MAE, MAPE, and R² across held-out test set

### Why HistGradientBoosting?
Unlike standard GradientBoosting, the histogram-based variant bins continuous features before fitting, making it significantly faster on large datasets (50K+ records) while achieving comparable or better accuracy.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python |
| ML Model | Scikit-learn (HistGradientBoosting) |
| Feature Engineering | Pandas, NumPy, PCA |
| Web App | Streamlit |
| PDF Generation | ReportLab |
| Experimentation | Jupyter Notebook, Google Colab |
| Version Control | Git, GitHub |

---

## Project Structure

```
compensiq/
├── app.py                  # Streamlit web application — UI, prediction flow, PDF export
├── compensiq.ipynb         # Full training notebook — EDA, feature engineering, model training & evaluation
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Jaswanth-ak/compensiq.git
cd compensiq
pip install -r requirements.txt
streamlit run app.py
```

App runs at `http://localhost:8501`

---

## Author

**Jaswanth B** — AI/ML Engineer

[GitHub](https://github.com/Jaswanth-ak) · [LinkedIn](https://linkedin.com/in/jaswanth-b-192676371) · jaswanthak46@gmail.com
