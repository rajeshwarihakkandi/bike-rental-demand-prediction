# Hourly Bike Rental Demand Forecasting with Ensemble Machine Learning

> Predicting bike-sharing demand by hour using weather, time, and seasonal signals — enabling operators to optimise fleet availability and cut customer wait times.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy)

---

## Executive Summary

### Business Problem

A bike-sharing operator running a 2-year fleet had no reliable way to predict hourly demand. Without forecasts, bikes pile up at low-demand stations during off-peak hours while high-demand stations run empty during rush hour — leading to lost revenue, customer frustration, and inefficient redistribution costs.

The core question: **How many bikes will be rented in a given hour, given the weather, day, and time?**

### The Solution

Built and tuned three tree-based regression models on 17,379 hourly rental records, comparing Decision Tree, Random Forest, and Gradient Boosting with 5-fold cross-validated hyperparameter search. The best model explains 95.4% of hourly demand variance with a mean error of just 23 rentals per hour. Deployed as an interactive Streamlit dashboard with real-time demand prediction.

### Number Impact

| Metric | Value |
|:---|:---|
| Dataset size | 17,379 hourly records (2011–2012) |
| Features used | 15 (after engineering) |
| Best model | Gradient Boosting (Tuned) |
| R² score | **0.9544** |
| RMSE | **37.98 rentals/hour** |
| MAE | **23.09 rentals/hour** |
| Variance explained | **95.4%** |

### Key Finding

Hour of day is the single strongest predictor of demand, far outweighing weather and seasonal factors. Weekday demand shows a clear bimodal commuter pattern (8 AM and 5–6 PM peaks), while weekend demand is flat and leisure-driven from 10 AM to 4 PM. A `rush_hour` flag and cyclic sine/cosine encoding of the hour feature were critical to capturing this pattern correctly.

### Next Steps

- Deploy the model as an API endpoint so fleet management software can pull hourly forecasts automatically
- Add real-time weather API integration to the dashboard for live predictions
- Extend to station-level forecasting — current model predicts system-wide demand, not per-station
- Retrain quarterly as seasonal patterns shift year-on-year

---

## Model Results

### Baseline vs Tuned Comparison

| Model | MAE (Baseline) | RMSE (Baseline) | R² (Baseline) | MAE (Tuned) | RMSE (Tuned) | R² (Tuned) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Decision Tree | 33.71 | 57.86 | 0.8943 | 30.73 | 50.79 | 0.9186 |
| Random Forest | 24.84 | 42.10 | 0.9440 | 25.26 | 41.22 | 0.9464 |
| **Gradient Boosting** | 39.67 | 56.77 | 0.8982 | **23.09** | **37.98** | **0.9544** |

> Gradient Boosting improved the most from tuning — baseline R² of 0.8982 jumped to 0.9544 after RandomizedSearchCV with 5-fold CV.

---

## Methodology

### Tech Stack

| Tool | Why It Was Used |
|:---|:---|
| **Pandas + NumPy** | Data cleaning, feature engineering, aggregations |
| **Scikit-Learn** | All three regression models + RandomizedSearchCV tuning |
| **StandardScaler** | Normalise cyclic features before modelling |
| **Matplotlib + Seaborn** | EDA visualisations (11 plots in notebook) |
| **Streamlit** | Interactive dashboard with real-time predictor |
| **Joblib** | Persist trained models to disk |

### Project Flow

```
Raw Dataset (17,379 rows · 17 columns)
         │
         ▼
Data Cleaning
  • Replace '?' placeholders with NaN
  • Median imputation for numeric columns
  • Mode imputation for categorical columns
  • Parse dteday to datetime
         │
         ▼
Encoding
  • season: springer/summer/fall/winter → 1/2/3/4
  • weathersit: Clear/Mist/Light Snow/Heavy Rain → 1/2/3/4
  • holiday / workingday: Yes/No → 1/0
  • Drop: instant (index), dteday (captured by yr/mnth/hr)
  • Drop: atemp (r=0.99 with temp — multicollinearity)
         │
         ▼
Feature Engineering
  rush_hour  = 1 if hr in [7, 8, 17, 18] else 0
  hr_sin     = sin(2pi x hr / 24)
  hr_cos     = cos(2pi x hr / 24)
  mnth_sin   = sin(2pi x mnth / 12)
  mnth_cos   = cos(2pi x mnth / 12)
  StandardScaler applied to cyclic features
         │
         ▼
Train / Test Split (80/20, random_state=42)
  Train: 13,903 rows
  Test:   3,476 rows
  Drop: casual + registered (sub-components of cnt — data leakage)
         │
         ▼
Baseline Models
  Decision Tree · Random Forest · Gradient Boosting
         │
         ▼
RandomizedSearchCV Tuning (5-fold CV, R² scoring)
  Decision Tree    : 30 iterations
  Random Forest    : 20 iterations
  Gradient Boosting: 20 iterations
         │
         ▼
Best Model Selected by R² — Gradient Boosting (Tuned)
R²=0.9544 · RMSE=37.98 · MAE=23.09
         │
         ▼
Streamlit Dashboard (6 pages)
  Overview · Trends · Weather · Model Comparison
  Live Prediction · Feature Interpretation
```

### Why Cyclic Encoding for Hour and Month

A raw hour column treated as an integer tells the model that hour 23 and hour 0 are 23 units apart. In reality they are 1 hour apart. Sine/cosine encoding wraps the scale correctly so the model sees midnight and 1 AM as close, not distant.

### Why Drop `casual` and `registered`

Both columns are sub-components of `cnt` (total rentals = casual + registered). Including them would give the model the answer before predicting — a textbook data leakage case.

---

## Key EDA Findings

**Hour of day drives demand more than any other feature.** Weekday hours show two sharp peaks at 8 AM and 5–6 PM (commuter pattern). Weekends show a flat leisure curve from 10 AM to 4 PM with no commuter spikes.

**Temperature has the strongest weather correlation (+0.40).** Humidity is negatively correlated (−0.32). Wind speed has minimal effect.

**2012 consistently outperforms 2011 in every month** — strong year-on-year growth confirms the `yr` feature must be included in the model.

**81.2% of rentals are by registered users.** Casual users are weather-dependent and leisure-driven. Registered users are consistent commuters with predictable hourly patterns.

---

## Dataset

**Source:** UCI Bike Sharing Dataset (hourly)
**Size:** 17,379 records · 17 attributes · 2011–2012

| Column | Description |
|:---|:---|
| season | Spring / Summer / Fall / Winter |
| yr | Year (2011 or 2012) |
| mnth | Month (1–12) |
| hr | Hour (0–23) |
| holiday | Whether a public holiday |
| weekday | Day of the week (0–6) |
| workingday | Working day or not |
| weathersit | Clear / Mist / Light Snow / Heavy Rain |
| temp | Normalised temperature |
| hum | Normalised humidity |
| windspeed | Normalised wind speed |
| casual | Count of casual users |
| registered | Count of registered users |
| cnt | Total rentals (target) |

---

## Project Structure

```
bike-rental-prediction/
│
├── Bike_rental_DS_group_3_.ipynb   # Full analysis notebook
├── app.py                          # Streamlit dashboard (6 pages)
├── train.py                        # Model training script
├── Dataset.csv                     # Raw dataset
├── requirements.txt                # Dependencies
└── README.md
```

> Model files (*.pkl) are not stored in the repo. Run `python train.py` to generate them.

---

## Setup

```bash
git clone https://github.com/rajeshwarihakkandi/bike-rental-prediction.git
cd bike-rental-prediction

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

# Generate model files
python train.py

# Launch dashboard
streamlit run app.py
```

---

## Author

**DS Group 3** — Presidency University, Bengaluru

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/rajeshwari-hakkandi)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/rajeshwarihakkandi)
