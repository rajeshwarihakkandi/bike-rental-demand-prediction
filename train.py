"""
train.py — Bike Rental Demand Prediction
Reproduces the full pipeline from the notebook:
  1. Load and clean Dataset.csv
  2. Feature engineering
  3. Train Decision Tree, Random Forest, Gradient Boosting (baseline)
  4. Tune all three with RandomizedSearchCV
  5. Evaluate and select best model by R2
  6. Save best_model.pkl, scaler.pkl, feature_names.pkl,
     all_metrics.pkl, test_preds.pkl,
     dt_tuned.pkl, rf_tuned.pkl, gb_tuned.pkl
Run: python train.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split, RandomizedSearchCV
from sklearn.preprocessing      import StandardScaler
from sklearn.tree               import DecisionTreeRegressor
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv('Dataset.csv')
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── 2. CLEAN DATA ─────────────────────────────────────────────────────────────

print("Cleaning data...")

for col in df.columns:
    mask = df[col].astype(str).str.strip() == '?'
    df.loc[mask, col] = np.nan

df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True, errors='coerce')

num_cols = ['yr', 'mnth', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

for c in ['temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth', 'casual', 'registered']:
    df[c].fillna(df[c].median(), inplace=True)
for c in ['season', 'holiday', 'workingday', 'weathersit']:
    df[c].fillna(df[c].mode()[0], inplace=True)

df['yr']         = df['yr'].round().astype('Int64')
df['mnth']       = df['mnth'].round().astype('Int64')
df['casual']     = df['casual'].round().astype('Int64')
df['registered'] = df['registered'].round().astype('Int64')

print(f"  Missing values after imputation: {df.isnull().sum().sum()}")

# ── 3. ENCODE CATEGORICALS ────────────────────────────────────────────────────

print("Encoding categorical features...")

season_map  = {'springer': 1, 'summer': 2, 'fall': 3, 'winter': 4}
weather_map = {'Clear': 1, 'Mist': 2, 'Light Snow': 3, 'Heavy Rain': 4}

df['season']     = df['season'].map(season_map)
df['weathersit'] = df['weathersit'].map(weather_map)
df['holiday']    = df['holiday'].map({'No': 0, 'Yes': 1})
df['workingday'] = df['workingday'].map({'No work': 0, 'Working Day': 1})
df['hr']         = df['hr'].astype(int)
df['instant']    = df['instant'].astype(int)

# ── 4. DROP UNNECESSARY COLUMNS ───────────────────────────────────────────────

drop_cols = ['instant', 'dteday', 'atemp']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 5. FEATURE ENGINEERING ────────────────────────────────────────────────────

print("Engineering features...")

df['rush_hour'] = df['hr'].isin([7, 8, 17, 18]).astype(int)
df['hr_sin']    = np.sin(2 * np.pi * df['hr'] / 24)
df['hr_cos']    = np.cos(2 * np.pi * df['hr'] / 24)
df['mnth_sin']  = np.sin(2 * np.pi * df['mnth'] / 12)
df['mnth_cos']  = np.cos(2 * np.pi * df['mnth'] / 12)

print(f"  Features after engineering: {df.shape[1]} columns")
df.dropna(inplace=True)
print(f"  Rows after dropping NaNs: {df.shape[0]:,}")

# ── 6. SCALE CYCLIC FEATURES ─────────────────────────────────────────────────

print("Scaling cyclic features...")

scale_cols = ['hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos']
scaler     = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ── 7. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────

print("Splitting data...")

X = df.drop(columns=['cnt', 'casual', 'registered'])
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"  Train: {X_train.shape[0]:,} rows")
print(f"  Test:  {X_test.shape[0]:,} rows")
print(f"  Features: {X_train.shape[1]}")

# ── 8. BASELINE MODELS ────────────────────────────────────────────────────────

print("\nTraining baseline models...")

baseline_models = {
    'Decision Tree'    : DecisionTreeRegressor(random_state=42),
    'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

baseline_results = {}
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    baseline_results[name] = {
        'MAE' : round(mean_absolute_error(y_test, preds), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        'R2'  : round(r2_score(y_test, preds), 4),
    }
    print(f"  {name:<25} R2={baseline_results[name]['R2']:.4f}  RMSE={baseline_results[name]['RMSE']:.2f}")

# ── 9. HYPERPARAMETER TUNING ──────────────────────────────────────────────────

print("\nTuning models with RandomizedSearchCV (this takes a few minutes)...")

dt_params = {
    'max_depth'        : [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 2, 5, 10],
    'max_features'     : ['sqrt', 'log2', None],
}
dt_search = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_distributions=dt_params,
    n_iter=30, cv=5, scoring='r2',
    n_jobs=-1, random_state=42, verbose=0
)
dt_search.fit(X_train, y_train)
print(f"  Decision Tree     best CV R2: {dt_search.best_score_:.4f}")

rf_params = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2'],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_params,
    n_iter=20, cv=5, scoring='r2',
    n_jobs=-1, random_state=42, verbose=0
)
rf_search.fit(X_train, y_train)
print(f"  Random Forest     best CV R2: {rf_search.best_score_:.4f}")

gb_params = {
    'n_estimators'     : [100, 200, 300],
    'learning_rate'    : [0.05, 0.1, 0.2],
    'max_depth'        : [3, 4, 5, 6],
    'subsample'        : [0.7, 0.8, 1.0],
    'min_samples_leaf' : [1, 2, 5],
}
gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gb_params,
    n_iter=20, cv=5, scoring='r2',
    n_jobs=-1, random_state=42, verbose=0
)
gb_search.fit(X_train, y_train)
print(f"  Gradient Boosting best CV R2: {gb_search.best_score_:.4f}")

# ── 10. EVALUATE TUNED MODELS ─────────────────────────────────────────────────

print("\nEvaluating tuned models on test set...")

tuned_models = {
    'Decision Tree (Tuned)'    : dt_search.best_estimator_,
    'Random Forest (Tuned)'    : rf_search.best_estimator_,
    'Gradient Boosting (Tuned)': gb_search.best_estimator_,
}

tuned_results = {}
preds_store   = {}

for name, model in tuned_models.items():
    preds = model.predict(X_test)
    preds_store[name] = preds
    tuned_results[name] = {
        'MAE' : round(mean_absolute_error(y_test, preds), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        'R2'  : round(r2_score(y_test, preds), 4),
    }
    print(f"  {name:<30} R2={tuned_results[name]['R2']:.4f}  RMSE={tuned_results[name]['RMSE']:.2f}  MAE={tuned_results[name]['MAE']:.2f}")

# ── 11. SELECT BEST MODEL ─────────────────────────────────────────────────────

tuned_df   = pd.DataFrame(tuned_results).T
best_name  = tuned_df['R2'].astype(float).idxmax()
best_model = tuned_models[best_name]
best_metrics = tuned_results[best_name]

print(f"\n  Best model : {best_name}")
print(f"  R2         : {best_metrics['R2']}")
print(f"  RMSE       : {best_metrics['RMSE']} rentals/hour")
print(f"  MAE        : {best_metrics['MAE']} rentals/hour")

# ── 12. SAVE ARTIFACTS ────────────────────────────────────────────────────────

print("\nSaving model artifacts...")

joblib.dump(best_model,              'best_model.pkl')
joblib.dump(scaler,                  'scaler.pkl')
joblib.dump(list(X.columns),         'feature_names.pkl')
joblib.dump(preds_store[best_name],  'test_preds.pkl')
joblib.dump(y_test.values,           'y_test.pkl')
joblib.dump(dt_search.best_estimator_, 'dt_tuned.pkl')
joblib.dump(rf_search.best_estimator_, 'rf_tuned.pkl')
joblib.dump(gb_search.best_estimator_, 'gb_tuned.pkl')

all_metrics = {
    'baseline': baseline_results,
    'tuned'   : tuned_results,
    'best'    : {
        'name': best_name,
        'MAE' : best_metrics['MAE'],
        'RMSE': best_metrics['RMSE'],
        'R2'  : best_metrics['R2'],
    }
}
joblib.dump(all_metrics, 'all_metrics.pkl')

print("  best_model.pkl      saved")
print("  scaler.pkl          saved")
print("  feature_names.pkl   saved")
print("  test_preds.pkl      saved")
print("  y_test.pkl          saved")
print("  dt_tuned.pkl        saved")
print("  rf_tuned.pkl        saved")
print("  gb_tuned.pkl        saved")
print("  all_metrics.pkl     saved")
print("\nDone. Run: streamlit run app.py")
