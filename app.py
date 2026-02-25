import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    layout="wide"
)

PROJECT_ROOT = Path(".")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"

# =====================================================
# LOAD CSS
# =====================================================

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

master_df = load_data()

# =====================================================
# SAFE LOAD MODEL
# =====================================================
def safe_load_model(path):
    try:
        if Path(path).exists():
            return joblib.load(path)
        else:
            print(f"File does not exist: {path}")
            return None
    except Exception as e:
        import traceback
        print(f"Error loading model: {path}")
        print(traceback.format_exc())
        return None

# =====================================================
# LOAD ALL MODELS
# =====================================================

regression_models = {
    "Linear Regression": safe_load_model(MODELS_DIR / "linear_regression.pkl"),
    "Random Forest Regression": safe_load_model(MODELS_DIR / "random_forest_regression.pkl"),
    "Gradient Boosting Regression": safe_load_model(MODELS_DIR / "gradient_boosting_regression.pkl"),
}

classification_models = {
    "Logistic Regression": safe_load_model(MODELS_DIR / "logistic_regression.pkl"),
    "Random Forest Classifier": safe_load_model(MODELS_DIR / "random_forest_clf.pkl"),
    "Gradient Boosting Classifier": safe_load_model(MODELS_DIR / "gradient_boosting_clf.pkl"),
    "XGBoost Classifier": safe_load_model(MODELS_DIR / "xgboost_clf.pkl"),
    "LightGBM Classifier": safe_load_model(MODELS_DIR / "lightgbm_clf.pkl"),
}

# =====================================================
# LOAD BEST MODEL FROM JSON
# =====================================================

def load_best_from_json(json_path):
    try:
        with open(json_path, "r") as f:
            config = json.load(f)

        model_path = Path(config["model_path"])
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        model = safe_load_model(model_path)

        if model is None:
            return None, None, None, False

        return model, config["model_name"], config["score"], True

    except:
        return None, None, None, False

best_reg_model, best_reg_name, best_reg_score, reg_status = load_best_from_json(
    REPORTS_DIR / "best_regression_model.json"
)

best_clf_model, best_clf_name, best_clf_score, clf_status = load_best_from_json(
    REPORTS_DIR / "best_classification_model.json"
)

# =====================================================
# OPTIONAL COMPONENTS
# =====================================================

scaler = safe_load_model(MODELS_DIR / "scaler.pkl")
label_encoders = safe_load_model(MODELS_DIR / "label_encoders.pkl")

# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

def preprocess_input(input_df, model):

    if model is None:
        return None

    df = input_df.copy()

    if label_encoders:
        for col in df.columns:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except:
                    df[col] = 0

    if hasattr(model, "feature_names_in_"):
        required = model.feature_names_in_
        aligned = pd.DataFrame(columns=required)
        aligned.loc[0] = 0

        for col in required:
            if col in df.columns:
                aligned.loc[0, col] = df[col].values[0]

        df = aligned

    if scaler:
        try:
            df = scaler.transform(df)
        except:
            pass

    return df

# =====================================================
# SIDEBAR (FIXED CARDS)
# =====================================================

def status_indicator(condition):
    if condition:
        return '<span style="color:#4CAF50;">● Loaded</span>'
    return '<span style="color:#FFC107;">● Missing</span>'

# Header Card
st.sidebar.markdown("""
<div class="sidebar-card">
<div class="sidebar-title">🌍 Tourism Analytics</div>
<div class="sidebar-sub">Advanced ML Control Center</div>
</div>
""", unsafe_allow_html=True)

# Model Selection Card
st.sidebar.markdown("""
<div class="sidebar-card">
<div class="sidebar-title">🧠 Model Selection</div>
<div class="sidebar-sub">Choose prediction strategy</div>
</div>
""", unsafe_allow_html=True)

selection_mode = st.sidebar.radio(
    "Mode",
    ["Automatic (Best)", "Manual"]
)

# Dataset Card (FIXED)
if master_df is not None:
    st.sidebar.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-title">📊 Dataset Overview</div>
        <div class="sidebar-sub" style="margin-top:10px;">
            • Records: <b>{master_df.shape[0]:,}</b><br>
            • Features: <b>{master_df.shape[1]}</b><br>
            • Unique Users: <b>{master_df['UserId'].nunique()}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">📊 Dataset Overview</div>
        <div style="color:#FF5252;margin-top:10px;">
            Dataset Missing
        </div>
    </div>
    """, unsafe_allow_html=True)

# System Status Card (FIXED)
st.sidebar.markdown(f"""
<div class="sidebar-card">
    <div class="sidebar-title">⚙ System Status</div>
    <div class="sidebar-sub" style="margin-top:10px;">
        Dataset: {status_indicator(master_df is not None)}<br>
        Scaler: {status_indicator(scaler)}<br>
        Encoders: {status_indicator(label_encoders)}<br>
        Best Regression: {status_indicator(best_reg_model)}<br>
        Best Classification: {status_indicator(best_clf_model)}
    </div>
</div>
""", unsafe_allow_html=True)

# Best Model Score Card
if reg_status or clf_status:
    st.sidebar.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-title">🏆 Best Model Scores</div>
        <div class="sidebar-sub" style="margin-top:10px;">
            {"Regression:<br><b>"+best_reg_name+"</b><br>Score: "+str(round(best_reg_score,4))+"<br><br>" if reg_status else ""}
            {"Classification:<br><b>"+best_clf_name+"</b><br>Score: "+str(round(best_clf_score,4)) if clf_status else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MAIN UI
# =====================================================

st.title("🌍 Tourism Experience Analytics Platform")

tab1, tab2, tab3 = st.tabs(
    ["📊 Rating Prediction", "🎯 Visit Mode Prediction", "⭐ Recommendation Engine"]
)


# =====================================================
# TAB 1 — REGRESSION
# =====================================================

with tab1:

    st.header("Predict Attraction Rating")

    active_reg_model = None
    active_reg_name = None

    if selection_mode == "Automatic (Best)":
        active_reg_model = best_reg_model
        active_reg_name = best_reg_name
    else:
        available = {k: v for k, v in regression_models.items() if v is not None}
        if available:
            selected = st.sidebar.selectbox(
                "Select Regression Model",
                list(available.keys()),
                key="reg_model_select"
            )
            active_reg_model = available[selected]
            active_reg_name = selected

    if active_reg_model and master_df is not None:

        col1, col2, col3 = st.columns(3)

        continent = col1.selectbox(
            "Continent",
            master_df["Continent"].unique(),
            key="reg_continent"
        )

        year = col2.selectbox(
            "Visit Year",
            sorted(master_df["VisitYear"].unique()),
            key="reg_year"
        )

        month = col3.selectbox(
            "Visit Month",
            sorted(master_df["VisitMonth"].unique()),
            key="reg_month"
        )

        if st.button("Predict Rating"):

            input_df = pd.DataFrame([{
                "Continent": continent,
                "VisitYear": year,
                "VisitMonth": month
            }])

            processed = preprocess_input(input_df, active_reg_model)

            if processed is not None:
                pred_scaled = active_reg_model.predict(processed)[0]

                if scaler:
                    rating_mean = scaler.mean_[0]
                    rating_std = scaler.scale_[0]
                    pred = (pred_scaled * rating_std) + rating_mean
                else:
                    pred = pred_scaled

                pred = max(1, min(5, pred))

                st.markdown(f"""
                <div class="card">
                <h3>Model Used: {active_reg_name}</h3>
                <h2>Predicted Rating: {round(pred,2)}</h2>
                </div>
                """, unsafe_allow_html=True)

# =====================================================
# TAB 2 — CLASSIFICATION
# =====================================================

with tab2:

    st.header("Predict Visit Mode")

    active_clf_model = None
    active_clf_name = None

    if selection_mode == "Automatic (Best)":
        active_clf_model = best_clf_model
        active_clf_name = best_clf_name
    else:
        available = {k: v for k, v in classification_models.items() if v is not None}
        if available:
            selected = st.sidebar.selectbox(
                "Select Classification Model",
                list(available.keys()),
                key="clf_model_select"
            )
            active_clf_model = available[selected]
            active_clf_name = selected

    if active_clf_model and master_df is not None:

        col1, col2, col3 = st.columns(3)

        continent = col1.selectbox(
            "Continent",
            master_df["Continent"].unique(),
            key="clf_continent"
        )

        year = col2.selectbox(
            "Visit Year",
            sorted(master_df["VisitYear"].unique()),
            key="clf_year"
        )

        month = col3.selectbox(
            "Visit Month",
            sorted(master_df["VisitMonth"].unique()),
            key="clf_month"
        )

        if st.button("Predict Visit Mode"):

            input_df = pd.DataFrame([{
                "Continent": continent,
                "VisitYear": year,
                "VisitMonth": month
            }])

            processed = preprocess_input(input_df, active_clf_model)

            if processed is not None:
               pred_encoded = active_clf_model.predict(processed)[0]

            # Decode class label if encoder exists
            pred = pred_encoded
            if label_encoders and "VisitMode" in label_encoders:
                try:
                    pred = label_encoders["VisitMode"].inverse_transform([pred_encoded])[0]
                except:
                    pred = pred_encoded

            confidence = None
            if hasattr(active_clf_model, "predict_proba"):
                confidence = active_clf_model.predict_proba(processed).max()

            st.markdown(f"""
            <div class="card">
            <h3>Model Used: {active_clf_name}</h3>
            <h2>Predicted Class: {pred}</h2>
            </div>
            """, unsafe_allow_html=True)

            if confidence:
                st.info(f"Confidence: {round(confidence*100,2)}%")

# =====================================================
# TAB 3 — RECOMMENDATION
# =====================================================

with tab3:

    st.header("Personalized Recommendations")

    if master_df is not None:

        user_id = st.selectbox(
            "Select User",
            master_df["UserId"].unique(),
            key="rec_user"
        )

        if st.button("Generate Recommendations"):

            user_data = master_df[master_df["UserId"] == user_id]

            if not user_data.empty:

                # System-defined number of recommendations
                SYSTEM_TOP_N = 5

                # Get user preference weights
                user_pref = user_data["AttractionType"].value_counts()

                # Score attractions based on user preference
                scored = master_df.merge(
                    user_pref.rename("weight"),
                    left_on="AttractionType",
                    right_index=True,
                    how="inner"
                )

                scored["final_score"] = scored["weight"]

                recs = (
                    scored.groupby("Attraction")["final_score"]
                    .sum()
                    .sort_values(ascending=False)
                )

                # Remove already visited attractions
                visited = user_data["Attraction"].unique()
                recs = recs[~recs.index.isin(visited)]

                # Auto-adjust if fewer items available
                final_recs = recs.head(SYSTEM_TOP_N)

                st.markdown(f"""
                <div class="card">
                <h3>Top Personalized Recommendations</h3>
                <p>Generated automatically based on user behavior</p>
                </div>
                """, unsafe_allow_html=True)

                for attraction, score in final_recs.items():
                    st.markdown(f"""
                    <div class="card">
                    <h4>{attraction}</h4>
                    <p>Recommendation Score: {score}</p>
                    </div>
                    """, unsafe_allow_html=True)
