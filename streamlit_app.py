import os
import sys
import json
import numpy as np

# Cross-version compatibility for NumPy pickle unpickling between NumPy 1.x and 2.x
if not hasattr(np, '_core'):
    try:
        import numpy.core as _np_core
        sys.modules['numpy._core'] = _np_core
        sys.modules['numpy._core.multiarray'] = _np_core.multiarray
        sys.modules['numpy._core.numeric'] = _np_core.numeric
    except Exception:
        pass
elif not hasattr(np, 'core'):
    try:
        import numpy._core as _np_core
        sys.modules['numpy.core'] = _np_core
        sys.modules['numpy.core.multiarray'] = _np_core.multiarray
        sys.modules['numpy.core.numeric'] = _np_core.numeric
    except Exception:
        pass

import joblib
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Train Delay Prediction - ML Mini Project",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# LOAD MODEL & ARTIFACTS (Cached for high performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, 'saved_model')
    
    with open(os.path.join(model_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    try:
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        encoders = joblib.load(os.path.join(model_dir, 'encoders.pkl'))
    except Exception as e:
        # Graceful fallback: If binary pickle differs across Cloud Python/NumPy versions,
        # train dynamically on clean fallback dataset sample (takes ~2 seconds)
        sample_path = os.path.join(base_dir, 'data', 'train_sample_small.csv')
        df_sample = pd.read_csv(sample_path)
        
        drop_cols = ['journey_id', 'train_number', 'departure_date', 'primary_delay_cause', 'delay_minutes', 'zone_abbr']
        df_clean = df_sample.drop(columns=[col for col in drop_cols if col in df_sample.columns])
        
        X = df_clean.drop(columns=['is_delayed'])
        y = df_clean['is_delayed']
        
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        
        encoders = {}
        for col in meta['cat_cols']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
        model.fit(X_scaled, y)
        
    return model, scaler, encoders, meta

model, scaler, encoders, meta = load_model_artifacts()

features = meta['features']
cat_mappings = meta['cat_mappings']
defaults = meta['feature_defaults']
metrics = meta['metrics']

# ------------------------------------------------------------------------------
# APP HEADER
# ------------------------------------------------------------------------------
st.title("🚆 Train Delay Prediction System")
st.markdown("**Machine Learning Mini Project** | Predicting whether a train will be delayed by >15 minutes before departure.")

# Metrics banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Production Model", "Gradient Boosting")
with col_m2:
    st.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
with col_m3:
    st.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
with col_m4:
    st.metric("Test F1-Score", f"{metrics['f1_score']:.3f}")

st.markdown("---")

# ------------------------------------------------------------------------------
# TABS SETUP
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Prediction", 
    "📊 Model Comparison (6 Models)", 
    "📈 Exploratory Data Analysis (EDA)", 
    "ℹ️ About & Responsible AI"
])

# ==============================================================================
# TAB 1: LIVE PREDICTION
# ==============================================================================
with tab1:
    st.subheader("Journey & Operational Inputs")
    st.markdown("Enter the train schedule and operational parameters below:")

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        train_type = st.selectbox(
            "Train Type", 
            options=cat_mappings['train_type'],
            index=cat_mappings['train_type'].index('Mail/Express') if 'Mail/Express' in cat_mappings['train_type'] else 0
        )
        season = st.selectbox(
            "Season / Weather", 
            options=cat_mappings['season'],
            index=cat_mappings['season'].index('Monsoon') if 'Monsoon' in cat_mappings['season'] else 0
        )
        zone = st.selectbox(
            "Railway Zone", 
            options=cat_mappings['zone'],
            index=cat_mappings['zone'].index('Northern Railway (NR)') if 'Northern Railway (NR)' in cat_mappings['zone'] else 0
        )
        departure_hour = st.slider("Departure Hour (24-Hour Format)", 0, 23, 14, help="14 = 2:00 PM")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            source_cat = st.selectbox("Source Station Tier", options=cat_mappings['source_station_category'], index=0)
        with col_s2:
            dest_cat = st.selectbox("Destination Station Tier", options=cat_mappings['destination_station_category'], index=2)

    with col_in2:
        distance_km = st.slider("Total Route Distance (km)", 20, 3000, 550, step=10)
        scheduled_hours = st.slider("Scheduled Travel Time (Hours)", 1.0, 50.0, 9.5, step=0.5)
        num_stops = st.slider("Number of Scheduled Halts", 1, 40, 12)
        ontime_pct = st.slider("Historical Route On-Time %", 10.0, 100.0, 65.0, step=1.0, help="Past punctuality on this route")

        st.markdown("**Operational Risk Factors:**")
        col_chk1, col_chk2 = st.columns(2)
        with col_chk1:
            late_rake = st.checkbox("Late Incoming Rake", value=False, help="Was the trainset already running late on its previous trip?")
            is_weekend = st.checkbox("Weekend Departure (Sat/Sun)", value=False)
        with col_chk2:
            track_doubled = st.checkbox("Doubled Track Route", value=True, help="Uncheck if running on single track")

    st.markdown("---")
    
    # Predict Button
    if st.button("🚀 Predict Train Delay Status", type="primary", use_container_width=True):
        input_data = defaults.copy()
        
        # Binary flags derived from inputs
        is_monsoon = 1 if season == 'Monsoon' else 0
        is_night = 1 if departure_hour in [22, 23, 0, 1, 2, 3, 4] else 0
        is_peak = 1 if departure_hour in [6, 7, 8, 17, 18, 19, 20] else 0

        # Update numerical inputs
        input_data['distance_km'] = distance_km
        input_data['scheduled_travel_hours'] = scheduled_hours
        input_data['num_scheduled_stops'] = num_stops
        input_data['departure_hour'] = departure_hour
        input_data['route_historical_ontime_pct'] = ontime_pct
        input_data['late_incoming_rake'] = 1 if late_rake else 0
        input_data['track_doubled'] = 1 if track_doubled else 0
        input_data['is_weekend'] = 1 if is_weekend else 0
        input_data['is_monsoon_season'] = is_monsoon
        input_data['is_night_departure'] = is_night
        input_data['is_peak_hour'] = is_peak
        
        # Weather severity proxy
        if season == 'Monsoon':
            input_data['season_severity_score'] = 0.85
        elif season == 'Winter/Fog':
            input_data['season_severity_score'] = 0.70
            input_data['is_fog_risk'] = 1
            input_data['fog_risk_score'] = 0.65
        else:
            input_data['season_severity_score'] = 0.35
            input_data['is_fog_risk'] = 0
            input_data['fog_risk_score'] = 0.0

        # Encode categorical variables
        input_data['train_type'] = int(encoders['train_type'].transform([train_type])[0])
        input_data['season'] = int(encoders['season'].transform([season])[0])
        input_data['zone'] = int(encoders['zone'].transform([zone])[0])
        input_data['source_station_category'] = int(encoders['source_station_category'].transform([source_cat])[0])
        input_data['destination_station_category'] = int(encoders['destination_station_category'].transform([dest_cat])[0])

        # DataFrame in exact feature order
        row_df = pd.DataFrame([input_data])[features]
        row_scaled = scaler.transform(row_df)

        # Run Prediction
        pred = model.predict(row_scaled)[0]
        prob_delay = model.predict_proba(row_scaled)[0][1] * 100

        # Display Result Card
        st.markdown("### Prediction Outcome")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if pred == 1:
                st.error("### ⚠️ DELAYED (>15 mins)")
                st.metric("Delay Probability", f"{prob_delay:.1f}%", delta="High Risk", delta_color="inverse")
            else:
                st.success("### ✅ ON-TIME (≤15 mins)")
                st.metric("On-Time Probability", f"{100 - prob_delay:.1f}%", delta="Normal", delta_color="normal")

        with col_res2:
            st.markdown(f"**Estimated Delay Probability: {prob_delay:.1f}%**")
            st.progress(prob_delay / 100.0)

            # Diagnostic risk factor notes
            st.markdown("**Key Factors Influencing This Prediction:**")
            factors = []
            if late_rake:
                factors.append("🚨 **Late Incoming Rake:** The trainset arrived late from its previous service.")
            if ontime_pct < 60:
                factors.append(f"📉 **Low Route Reliability:** Historical on-time punctuality is only {ontime_pct:.1f}%.")
            if season in ['Monsoon', 'Winter/Fog']:
                factors.append(f"🌧️ **Adverse Weather Risk:** Elevated seasonal severity index for {season}.")
            if not track_doubled:
                factors.append("⚠️ **Single Track Route:** Crossing and siding halts increase delay probability.")
            if is_peak:
                factors.append("🚦 **Peak Hour Departure:** High corridor traffic congestion.")
            if not factors:
                factors.append("✅ **Normal Operating Conditions:** Regular schedule adherence expected.")

            for f in factors:
                st.markdown(f"- {f}")

# ==============================================================================
# TAB 2: MODEL COMPARISON (6 MODELS)
# ==============================================================================
with tab2:
    st.subheader("Model Evaluation & Comparison (6 Models)")
    st.markdown("All 6 models were trained on preprocessed data and evaluated on an independent test set:")

    comparison_data = [
        {"Model": "Gradient Boosting", "Category": "Ensemble (Boosting)", "Accuracy": "85.20%", "Precision": "87.57%", "Recall": "92.65%", "F1-Score": "0.9004", "ROC-AUC": "0.9122", "Status": "⭐ Deployed Champion"},
        {"Model": "Logistic Regression", "Category": "Individual (Linear)", "Accuracy": "85.65%", "Precision": "88.51%", "Recall": "92.07%", "F1-Score": "0.9026", "ROC-AUC": "0.9154", "Status": "Baseline"},
        {"Model": "AdaBoost", "Category": "Ensemble (Boosting)", "Accuracy": "84.66%", "Precision": "87.85%", "Recall": "91.39%", "F1-Score": "0.8958", "ROC-AUC": "0.9084", "Status": "Ensemble 2"},
        {"Model": "Decision Tree", "Category": "Individual (Tree)", "Accuracy": "81.82%", "Precision": "85.85%", "Recall": "89.58%", "F1-Score": "0.8767", "ROC-AUC": "0.8563", "Status": "Baseline"},
        {"Model": "K-Nearest Neighbors", "Category": "Individual (Instance)", "Accuracy": "80.52%", "Precision": "85.33%", "Recall": "88.17%", "F1-Score": "0.8673", "ROC-AUC": "0.8404", "Status": "Baseline"},
        {"Model": "Gaussian Naive Bayes", "Category": "Individual (Probabilistic)", "Accuracy": "73.72%", "Precision": "92.28%", "Recall": "69.40%", "F1-Score": "0.7922", "ROC-AUC": "0.8654", "Status": "Baseline"}
    ]
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        if os.path.exists("plots/roc_curves.png"):
            st.image("plots/roc_curves.png", caption="ROC Curves Comparison (AUC > 0.91 for Boosting & Logistic)", use_container_width=True)
    with col_plot2:
        if os.path.exists("plots/model_comparison_barchart.png"):
            st.image("plots/model_comparison_barchart.png", caption="Accuracy & F1-Score Across Models", use_container_width=True)

    if os.path.exists("plots/confusion_matrices.png"):
        st.image("plots/confusion_matrices.png", caption="Confusion Matrices for All 6 Models", use_container_width=True)

# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab3:
    st.subheader("Exploratory Data Analysis Findings")
    st.markdown("Insights gathered from analyzing historical passenger train journey records:")

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        if os.path.exists("plots/target_distribution.png"):
            st.image("plots/target_distribution.png", caption="Class Distribution: 72% Delayed vs 28% On-Time", use_container_width=True)
        if os.path.exists("plots/delay_by_season.png"):
            st.image("plots/delay_by_season.png", caption="Delay Rate Across Seasons (Monsoon & Winter lead)", use_container_width=True)
            
    with col_eda2:
        if os.path.exists("plots/delay_by_train_type.png"):
            st.image("plots/delay_by_train_type.png", caption="Delays by Train Type (Vande Bharat 44% vs Passenger 88%)", use_container_width=True)
        if os.path.exists("plots/feature_importance.png"):
            st.image("plots/feature_importance.png", caption="Top Predictors in Gradient Boosting Model", use_container_width=True)

# ==============================================================================
# TAB 4: ABOUT & RESPONSIBLE AI
# ==============================================================================
with tab4:
    st.subheader("About the Project")
    st.markdown("""
    - **Academic Level:** Machine Learning Mini Project
    - **Team Size:** 2 Members
    - **Core Problem:** Binary Classification predicting train arrival delay exceeding 15 minutes before train departure.
    - **Ensemble Requirement:** Shortlisted 6 models including 2 boosting ensembles (**AdaBoost** and **Gradient Boosting**, excluding Random Forest).
    
    ---
    ### Responsible Use of AI Tools
    - **Declaration of AI Usage:** We used AI as a coding assistant to explore datasets, suggest baseline algorithms, write boilerplate preprocessing code, and debug syntax.
    - **Critical Evaluation:** We audited all AI outputs, removed post-trip leakage features (`delay_minutes`), adjusted sample sizes to prevent KNN memory stalls, and replaced Random Forest with boosting methods.
    - **Design Ownership:** The problem threshold (15 minutes), model shortlists, parameter choices, and report conclusions were directed and verified by the student team.
    """)

st.markdown("---")
st.caption("Machine Learning Mini Project © 2026 | Built with Streamlit, Python & Scikit-Learn")
