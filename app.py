"""
Bike Rental Demand Forecaster — Streamlit App
Mirrors the exact preprocessing pipeline from the notebook:
  Label encode → Drop cols → Rush hour + Cyclic features → StandardScaler (cyclic cols only)

Run:  streamlit run app.py
Needs: Dataset.csv + all .pkl files in the same folder (no sub‑folder)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Demand Forecaster",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_theme(style="whitegrid")
PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SEASON_MAP   = {'springer': 1, 'summer': 2, 'fall': 3, 'winter': 4}
WEATHER_MAP  = {'Clear': 1, 'Mist': 2, 'Light Snow': 3, 'Heavy Rain': 4}
MONTH_MAP    = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
WEEKDAY_MAP  = {0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
SEASON_ORDER = ['Spring','Summer','Fall','Winter']
SCALE_COLS   = ['hr_sin','hr_cos','mnth_sin','mnth_cos']

# ─── LOAD DATA (assume Dataset.csv in the same folder) ──────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("Dataset.csv")
    except FileNotFoundError:
        st.error("Dataset.csv not found. Place it in the same folder as app.py.")
        st.stop()

    for col in df.columns:
        mask = df[col].astype(str).str.strip() == "?"
        df.loc[mask, col] = np.nan

    df["dteday"] = pd.to_datetime(df["dteday"], dayfirst=True, errors="coerce")
    for c in ['yr','mnth','temp','atemp','hum','windspeed','casual','registered']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['temp','atemp','hum','windspeed','yr','mnth','casual','registered']:
        df[c].fillna(df[c].median(), inplace=True)
    for c in ['season','holiday','workingday','weathersit']:
        df[c].fillna(df[c].mode()[0], inplace=True)

    df['yr']         = df['yr'].round().astype('Int64')
    df['mnth']       = df['mnth'].round().astype('Int64')
    df['casual']     = df['casual'].round().astype('Int64')
    df['registered'] = df['registered'].round().astype('Int64')

    df['month_label']   = df['mnth'].map(MONTH_MAP)
    df['weekday_label'] = df['weekday'].map(WEEKDAY_MAP)
    df['year_label']    = df['yr'].astype(float).round().astype('Int64').map({2011:'2011',2012:'2012'})
    df['season_label']  = (df['season'].astype(str).str.strip().str.title()
                           .replace({'Springer':'Spring'}))
    return df

# ─── LOAD MODELS (assume all .pkl files are in the same folder) ──────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        best     = joblib.load("best_model.pkl")
        scaler   = joblib.load("scaler.pkl")
        features = joblib.load("feature_names.pkl")
        metrics  = joblib.load("all_metrics.pkl")
        test     = joblib.load("test_preds.pkl")
        # also load tuned versions for comparison if available
        dt = None
        rf = None
        gb = None
        for f in ["dt_tuned.pkl", "rf_tuned.pkl", "gb_tuned.pkl"]:
            try:
                if f == "dt_tuned.pkl":
                    dt = joblib.load(f)
                elif f == "rf_tuned.pkl":
                    rf = joblib.load(f)
                elif f == "gb_tuned.pkl":
                    gb = joblib.load(f)
            except:
                pass
        return {"best":best, "scaler":scaler, "features":features,
                "metrics":metrics, "test":test, "dt":dt, "rf":rf, "gb":gb}
    except Exception as e:
        st.error(f"Could not load model files. Ensure all .pkl files are in the same folder as app.py.\nError: {e}")
        st.stop()

df  = load_data()
mdl = load_models()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚲 Bike Demand")
    st.caption("DS Group 3 — Demand Forecaster")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Overview",
        "📈  Trends & Seasonality",
        "🌦️  Weather & Patterns",
        "🤖  Model Comparison",
        "🔮  Live Prediction",
        "💡  Interpretation",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Dataset: 17,379 records · 2011–2012\n\nModels: Decision Tree · Random Forest · Gradient Boosting")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.title("🚲 Bike Rental Demand Forecasting")
    st.markdown(
        "Predicting hourly demand in urban bike-sharing systems using weather, "
        "time, and seasonal signals — so operators can put the right number of "
        "bikes at the right dock before demand arrives."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",    f"{len(df):,}")
    c2.metric("Total Rentals",    f"{int(df['cnt'].sum()):,}")
    c3.metric("Peak Hour Demand", f"{int(df['cnt'].max())}")
    c4.metric("Registered Share", f"{df['registered'].sum()/df['cnt'].sum()*100:.1f}%")
    c5.metric("Casual Share",     f"{df['casual'].sum()/df['cnt'].sum()*100:.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns([2.2, 1])

    with col_l:
        st.subheader("Daily Rental Volume — 2011 to 2012")
        daily = df.groupby("dteday")["cnt"].sum()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(daily.index, daily.values, lw=0.8, color=PALETTE[0], alpha=0.7)
        ax.fill_between(daily.index, daily.values, alpha=0.12, color=PALETTE[0])
        ax.plot(daily.index, daily.rolling(15).mean().values,
                lw=2.2, color=PALETTE[1], label="15-day MA")
        ax.set_ylabel("Daily Rentals")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("User Mix by Season")
        sdf = (df[df["season_label"].isin(SEASON_ORDER)]
               .groupby("season_label")[["casual","registered"]]
               .mean().reindex(SEASON_ORDER))
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sdf.plot(kind="bar", stacked=True, ax=ax,
                 color=["#f4a460","#4169e1"], width=0.55)
        ax.set_xlabel(""); ax.set_ylabel("Avg Count")
        ax.tick_params(axis="x", rotation=0); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(
        df[["dteday","hr","season_label","temp","hum","windspeed","weathersit","cnt"]].head(10),
        use_container_width=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRENDS & SEASONALITY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈  Trends & Seasonality":
    st.title("📈 Trends & Seasonality")

    st.subheader("Year-over-Year Monthly Comparison")
    yoy = df.groupby(["yr","mnth"])["cnt"].mean().reset_index()
    yoy["year_label"] = yoy["yr"].astype(float).round().astype("Int64").map({2011:"2011",2012:"2012"})
    fig, ax = plt.subplots(figsize=(12, 4))
    for lbl, grp in yoy.groupby("year_label"):
        c = PALETTE[0] if lbl == "2011" else PALETTE[1]
        ax.plot(grp["mnth"], grp["cnt"], marker="o", lw=2.2, label=lbl, color=c)
        ax.fill_between(grp["mnth"], grp["cnt"], alpha=0.08, color=c)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(list(MONTH_MAP.values()))
    ax.set_ylabel("Avg Hourly Rentals"); ax.legend(title="Year")
    ax.grid(axis="y", alpha=0.4); fig.tight_layout()
    st.pyplot(fig); plt.close()

    yoy_w = yoy.pivot(index="mnth", columns="year_label", values="cnt")
    if "2011" in yoy_w.columns and "2012" in yoy_w.columns:
        yoy_w["Growth (%)"] = ((yoy_w["2012"]-yoy_w["2011"])/yoy_w["2011"]*100).round(1)
        yoy_w.index = list(MONTH_MAP.values())
        st.caption("Monthly growth 2011 → 2012")
        st.dataframe(yoy_w.style.format("{:.1f}"), use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Avg Rentals by Season")
        s_df = (df[df["season_label"].isin(SEASON_ORDER)]
                .groupby("season_label")["cnt"].mean()
                .reindex(SEASON_ORDER).reset_index())
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(data=s_df, x="season_label", y="cnt",
                    palette="Set2", order=SEASON_ORDER, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel("Avg Count")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Avg Rentals by Month")
        m_df = df.groupby("month_label")["cnt"].mean().reindex(list(MONTH_MAP.values())).reset_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(data=m_df, x="month_label", y="cnt",
                    palette="Blues_d", order=list(MONTH_MAP.values()), ax=ax)
        ax.set_xlabel(""); ax.set_ylabel("Avg Count")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Hour × Day of Week Demand Heatmap")
    pivot = (df.pivot_table(values="cnt", index="hr",
                            columns="weekday_label", aggfunc="mean")
             [["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]])
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, ax=ax,
                annot=True, fmt=".0f", annot_kws={"size": 7},
                cbar_kws={"label": "Avg Rentals"})
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Hour of Day")
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.caption("Commuter double-peak on weekdays (8 AM & 5 PM). "
               "Weekend demand spreads evenly 10 AM–4 PM — leisure riding.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WEATHER & PATTERNS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌦️  Weather & Patterns":
    st.title("🌦️ Weather & Behavioural Patterns")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Rentals by Weather Condition")
        w_df = (df.groupby("weathersit")["cnt"].mean()
                  .reset_index().sort_values("cnt", ascending=False))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(w_df["weathersit"].astype(str), w_df["cnt"],
                      color=["#2ecc71","#f39c12","#e74c3c","#8e44ad"][:len(w_df)])
        ax.set_xlabel("Weather Situation"); ax.set_ylabel("Avg Hourly Rentals")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Casual vs Registered — By Hour")
        hr_s = df.groupby("hr")[["casual","registered"]].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(hr_s.index, hr_s["casual"],
                label="Casual", marker="o", color="#f4a460", lw=2, ms=4)
        ax.plot(hr_s.index, hr_s["registered"],
                label="Registered", marker="s", color="#4169e1", lw=2, ms=4)
        ax.set_xticks(range(0,24)); ax.set_xlabel("Hour (0–23)")
        ax.set_ylabel("Avg Count"); ax.legend()
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)
    sample = df.sample(min(3000, len(df)), random_state=42)

    with col_l2:
        st.subheader("Rentals vs Temperature")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(sample["temp"], sample["cnt"], alpha=0.15, color="tomato", s=5)
        ax.set_xlabel("Normalized Temperature"); ax.set_ylabel("Rentals")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r2:
        st.subheader("Rentals vs Humidity")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(sample["hum"], sample["cnt"], alpha=0.15, color="teal", s=5)
        ax.set_xlabel("Normalized Humidity"); ax.set_ylabel("Rentals")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Holiday vs Non-Holiday")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=df, x="holiday", y="cnt", palette="Set1", ax=ax,
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with col_b:
        st.subheader("Working Day vs Non-Working Day")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=df, x="workingday", y="cnt", palette="Set2", ax=ax,
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Comparison":
    st.title("🤖 Model Comparison")

    if "metrics" not in mdl:
        st.warning("Run the notebook and save models first (`all_metrics.pkl` not found).")
        st.stop()

    baseline  = pd.DataFrame(mdl["metrics"]["baseline"]).T
    tuned     = pd.DataFrame(mdl["metrics"]["tuned"]).T
    best_name = tuned["R2"].idxmax()

    st.subheader("Results Table")
    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Baseline models")
        st.dataframe(baseline.style.format("{:.4f}"), use_container_width=True)
    with col_r:
        st.caption("After hyperparameter tuning (best model highlighted)")
        def hl(row):
            return ["background-color:#d4edda"]*len(row) \
                   if row.name == best_name else [""]*len(row)
        st.dataframe(tuned.style.format("{:.4f}").apply(hl, axis=1),
                     use_container_width=True)
    st.caption(f"✅ Best model: **{best_name}**")

    st.markdown("---")
    st.subheader("Visual Comparison — Baseline vs Tuned")

    all_df = pd.concat([
        baseline.assign(Type="Baseline"),
        tuned.assign(Type="Tuned")
    ]).reset_index().rename(columns={"index": "Model"})
    all_df["Short"] = (all_df["Model"]
                       .str.replace(" (Tuned)", "", regex=False)
                       .str.replace("Gradient Boosting","GBR")
                       .str.replace("Random Forest","RF")
                       .str.replace("Decision Tree","DT"))
    color_map = {"Baseline":"#aec6cf","Tuned":"#2ecc71"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Baseline vs Tuned — All Models", fontsize=13, fontweight="bold")
    for ax, metric in zip(axes, ["MAE","RMSE","R2"]):
        for i, (_, row) in enumerate(all_df.iterrows()):
            ax.bar(i, row[metric], color=color_map[row["Type"]],
                   edgecolor="white", width=0.6)
        ax.set_xticks(range(len(all_df)))
        ax.set_xticklabels([f"{r['Short']}\n({r['Type']})"
                            for _, r in all_df.iterrows()],
                           fontsize=7, rotation=25)
        ax.set_title(metric); ax.set_ylabel(metric)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=v, label=k) for k,v in color_map.items()], fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    if "test" in mdl:
        st.subheader(f"Diagnostic Plots — {mdl['metrics']['best']['name']}")
        y_test    = joblib.load("y_test.pkl")
        y_pred    = mdl["test"]
        residuals = y_test - y_pred

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Actual vs Predicted (500 samples)**")
            rng = np.random.default_rng(42)
            idx = rng.choice(len(y_test), min(500, len(y_test)), replace=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test[idx], y_pred[idx], alpha=0.3, s=10, color="steelblue")
            lims = [0, max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.legend(fontsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            st.markdown("**Residual Distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(residuals, bins=50, color="coral", edgecolor="white", linewidth=0.4)
            ax.axvline(0, color="black", lw=1.5, linestyle="--")
            ax.set_xlabel("Residual (Actual − Predicted)"); ax.set_ylabel("Frequency")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae  = float(np.mean(np.abs(residuals)))
        r2   = float(1 - np.sum(residuals**2)/np.sum((y_test - y_test.mean())**2))
        mask_nz = y_test > 0
        mape = float(np.mean(np.abs(residuals[mask_nz]/y_test[mask_nz]))*100)
        under_rate = float((y_pred < y_test).sum()/len(y_test)*100)

        st.markdown("---")
        st.subheader("Business Metrics")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("MAE",   f"{mae:.2f}")
        b2.metric("RMSE",  f"{rmse:.2f}")
        b3.metric("R²",    f"{r2:.4f}")
        b4.metric("MAPE",  f"{mape:.1f}%")
        b5.metric("Underprediction %", f"{under_rate:.1f}%",
                  help="Hours where predicted < actual → empty-dock risk")

    if "best" in mdl and "features" in mdl:
        st.markdown("---")
        st.subheader("Feature Importance — Best Model")
        fi = pd.Series(mdl["best"].feature_importances_, index=mdl["features"])
        fi = fi.sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=fi.values, y=fi.index, palette="viridis", ax=ax)
        ax.set_xlabel("Importance Score")
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Live Prediction":
    st.title("🔮 Live Demand Prediction")
    st.markdown("Set the conditions and get an instant forecast for that hour.")

    if "best" not in mdl:
        st.warning("Run the notebook to generate `best_model.pkl`.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📅 Time")
        hour     = st.slider("Hour of day", 0, 23, 8)
        weekday  = st.selectbox("Day of week",
                                options=list(WEEKDAY_MAP.keys()),
                                format_func=lambda x: WEEKDAY_MAP[x], index=1)
        mnth     = st.selectbox("Month",
                                options=list(MONTH_MAP.keys()),
                                format_func=lambda x: MONTH_MAP[x], index=5)
        yr       = st.selectbox("Year", [2011, 2012], index=1)

    with col2:
        st.subheader("🌤️ Weather")
        temp           = st.slider("Temperature (normalised 0–1)", 0.0, 1.0, 0.50, 0.01)
        hum            = st.slider("Humidity (normalised 0–1)",    0.0, 1.0, 0.63, 0.01)
        windspeed      = st.slider("Wind speed (normalised 0–1)",  0.0, 0.85, 0.19, 0.01)
        weather_choice = st.selectbox("Weather situation", options=list(WEATHER_MAP.keys()))

    with col3:
        st.subheader("📆 Context")
        season_choice  = st.selectbox("Season",
                                      options=list(SEASON_MAP.keys()),
                                      format_func=str.capitalize, index=1)
        holiday_choice = st.selectbox("Holiday?", ["No","Yes"])
        working_choice = st.selectbox("Working day?", ["Working Day","No work"])

    st.markdown("---")

    if st.button("🚲 Predict Demand", type="primary", use_container_width=True):
        # Build feature row — exactly mirroring the notebook pipeline
        row = {
            "yr"        : int(yr),
            "mnth"      : int(mnth),
            "hr"        : int(hour),
            "weekday"   : int(weekday),
            "temp"      : float(temp),
            "hum"       : float(hum),
            "windspeed" : float(windspeed),
            "season"    : SEASON_MAP[season_choice],
            "weathersit": WEATHER_MAP[weather_choice],
            "holiday"   : 1 if holiday_choice == "Yes" else 0,
            "workingday": 1 if working_choice == "Working Day" else 0,
        }
        # Cyclic + rush_hour features (notebook cell 71)
        row["rush_hour"] = int(hour in [7, 8, 17, 18])
        row["hr_sin"]    = np.sin(2 * np.pi * hour / 24)
        row["hr_cos"]    = np.cos(2 * np.pi * hour / 24)
        row["mnth_sin"]  = np.sin(2 * np.pi * int(mnth) / 12)
        row["mnth_cos"]  = np.cos(2 * np.pi * int(mnth) / 12)

        row_df = pd.DataFrame([row])

        # Scale cyclic cols (notebook cell 74)
        row_df[SCALE_COLS] = mdl["scaler"].transform(row_df[SCALE_COLS])

        # Align to training column order
        feat_cols = mdl["features"]
        for col in feat_cols:
            if col not in row_df.columns:
                row_df[col] = 0
        row_df = row_df[feat_cols]

        pred = max(0, int(round(mdl["best"].predict(row_df)[0])))

        # Historical context
        hist     = df[(df["hr"] == hour) & (df["weekday"] == weekday)]
        hist_avg = int(hist["cnt"].mean()) if len(hist) else pred
        hist_max = int(hist["cnt"].max())  if len(hist) else pred
        delta    = pred - hist_avg

        c_a, c_b, c_c, c_d = st.columns(4)
        c_a.metric("Predicted Rentals",        f"{pred}",
                   delta=f"{delta:+d} vs avg", delta_color="normal")
        c_b.metric("Historical avg (hr & day)",f"{hist_avg}")
        c_c.metric("Historical max (hr & day)",f"{hist_max}")
        c_d.metric("Rush hour?", "⚡ Yes" if row["rush_hour"] else "No")

        fig, ax = plt.subplots(figsize=(7, 2))
        ax.barh(["Predicted","Hist. Max"],
                [pred, hist_max],
                color=[PALETTE[2] if pred/max(hist_max,1) < 0.75 else PALETTE[1],
                       "#dddddd"], height=0.35)
        ax.axvline(hist_avg, color="royalblue", lw=2, linestyle="--",
                   label=f"Hist. avg = {hist_avg}")
        ax.set_xlim(0, max(hist_max, pred) * 1.12)
        ax.legend(fontsize=8); ax.set_xlabel("Bike rentals")
        fig.tight_layout(); st.pyplot(fig); plt.close()

        ratio = pred / max(hist_max, 1)
        if pred >= 600 or ratio > 0.85:
            st.error(f"🔴 Very high demand ({pred} bikes). "
                     "Ensure full dock capacity and fast rebalancing.")
        elif pred >= 300 or ratio > 0.60:
            st.warning(f"🟡 Moderate-high demand ({pred} bikes). "
                       "Consider pre-positioning extra bikes.")
        elif pred >= 100:
            st.success(f"🟢 Normal demand ({pred} bikes). Routine operations.")
        else:
            st.info(f"🔵 Low demand ({pred} bikes). "
                    "Good window for maintenance or redistribution.")

        with st.expander("Show input sent to model"):
            st.dataframe(row_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💡  Interpretation":
    st.title("💡 Model Interpretation")

    if "best" in mdl and "features" in mdl:
        fi = pd.Series(mdl["best"].feature_importances_, index=mdl["features"])
        fi = fi.sort_values(ascending=False)

        st.subheader("Feature Importance — Top 15")
        engineered = {"rush_hour","hr_sin","hr_cos","mnth_sin","mnth_cos"}
        colors_fi  = ["#e74c3c" if f in engineered else PALETTE[0]
                      for f in fi.head(15).index]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(fi.head(15).index[::-1], fi.head(15).values[::-1],
                color=colors_fi[::-1], edgecolor="white")
        ax.set_xlabel("Importance Score")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#e74c3c", label="Engineered feature"),
            Patch(color=PALETTE[0],label="Original feature"),
        ], fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        eng_df = fi[fi.index.isin(engineered)].reset_index()
        eng_df.columns = ["Feature","Importance"]
        eng_df["Global Rank"] = fi.rank(ascending=False).reindex(
            eng_df["Feature"]).values.astype(int)
        st.caption("Engineered features and their overall importance rank:")
        st.dataframe(eng_df.sort_values("Global Rank")
                     .style.format({"Importance":"{:.4f}"}),
                     use_container_width=True)
    else:
        st.info("Run the notebook to generate saved model files.")

    st.markdown("---")
    st.subheader("Feature Explanations")

    explanations = {
        "hr  (Hour of day)": (
            "The strongest single predictor. Demand follows a rigid daily rhythm — "
            "near zero from midnight to 5 AM, sharp twin spikes at 8 AM and 5–6 PM "
            "on weekdays (commuters), a smoother midday curve on weekends (leisure). "
            "No model can forecast demand well without this feature."
        ),
        "hr_sin / hr_cos  (Cyclic hour encoding)": (
            "Raw hour as an integer treats 23 and 0 as far apart — they're actually "
            "adjacent. Sine/cosine encoding wraps the clock into a circle so the model "
            "correctly understands that hour 23 is one step from hour 0."
        ),
        "yr  (Year)": (
            "Captures system-wide growth — 2012 rentals were ~40% higher than 2011 "
            "in every single month. Without this, the model would systematically "
            "under-predict demand for any year after the training baseline."
        ),
        "temp  (Normalised temperature)": (
            "Warmer weather means more riders, up to a comfort plateau around 0.6–0.7 "
            "(roughly 25–30 C). Below ~10 C demand drops sharply. "
            "One of the top non-time predictors."
        ),
        "rush_hour  (Engineered flag)": (
            "Binary: 1 if hour in {7, 8, 17, 18}, else 0. "
            "Gives the model a direct shortcut to the commuter peaks identified in "
            "EDA Plot 6, rather than learning them slowly from hr alone."
        ),
        "mnth_sin / mnth_cos  (Cyclic month encoding)": (
            "Like the hour encoding, these wrap December back to January. "
            "Seasonal demand is a smooth cycle — December is closer to January "
            "than it is to June."
        ),
        "hum  (Humidity)": (
            "Negatively correlated. Humidity above 0.8 suppresses rentals even "
            "without visible rain — damp air discourages casual riders."
        ),
        "season / weathersit": (
            "Fall and clear weather drive peak demand. "
            "Heavy rain or snow can push hourly demand to near zero — "
            "these conditions define the operational floor for dock planning."
        ),
        "weekday": (
            "Separates the commuter-heavy weekday rhythm from the leisure-dominated "
            "weekend profile. The same hour means completely different behaviour "
            "on a Monday vs a Saturday."
        ),
    }

    for feat, text in explanations.items():
        with st.expander(f"**{feat}**"):
            st.write(text)

    st.markdown("---")
    st.subheader("Operational Recommendations")
    st.markdown("""
| Finding | Action |
|---|---|
| Bimodal 8 AM & 5 PM peaks on weekdays | Pre-position bikes 30 min before rush windows |
| Fall highest, Winter lowest | Scale fleet seasonally — reduce in Jan–Feb |
| 2012 demand ~40% above 2011 | Plan for continued year-on-year fleet growth |
| Heavy rain drops demand >70% | Use rain forecasts to trigger same-day redistribution |
| Casual peak: midday weekends | Keep leisure routes stocked Sat–Sun 10 AM–4 PM |
| High humidity suppresses demand | Factor humidity into next-day supply decisions |
| Registered = 81% of demand | Prioritise subscription retention over casual acquisition |
    """)