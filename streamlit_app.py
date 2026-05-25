# ============================================================
# ABALONE AGE CLASSIFICATION STREAMLIT APP
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Abalone Age Classification",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DARK GRADIENT UI DESIGN
# ============================================================

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: radial-gradient(circle at top left, #164e63 0%, #0f172a 35%, #020617 100%);
    color: #f8fafc;
}

/* Sidebar Background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 50%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Main Headings */
h1, h2, h3 {
    color: #38bdf8 !important;
    font-weight: 800 !important;
}

/* Paragraph Text */
p, li, label, span {
    color: #e5e7eb;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.80);
    border: 1px solid rgba(56, 189, 248, 0.30);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #2563eb);
    color: white;
    border: none;
    border-radius: 14px;
    height: 3.2em;
    width: 100%;
    font-weight: 700;
    font-size: 16px;
    transition: 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #22d3ee, #3b82f6);
    transform: scale(1.01);
}

/* Download Button */
.stDownloadButton > button {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white;
    border: none;
    border-radius: 14px;
    height: 3em;
    font-weight: 700;
}

/* Input Widgets */
.stNumberInput input, .stSelectbox div {
    border-radius: 10px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 700;
    color: #e5e7eb;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Info Boxes */
.stAlert {
    border-radius: 14px;
}

/* Custom Hero Card */
.hero-card {
    padding: 34px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(6,182,212,0.20), rgba(37,99,235,0.16));
    border: 1px solid rgba(56,189,248,0.32);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 44px;
    font-weight: 900;
    color: white;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    line-height: 1.6;
}

/* Glass Cards */
.glass-card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    margin-bottom: 18px;
}

/* Prediction Label */
.prediction-box {
    padding: 24px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(59,130,246,0.18));
    border: 1px solid rgba(34,197,94,0.30);
    text-align: center;
    margin-top: 15px;
}

.prediction-main {
    font-size: 28px;
    font-weight: 900;
    color: #bbf7d0;
}

.prediction-sub {
    font-size: 17px;
    color: #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_DIR = "models"

HIDDEN_COLUMNS = [
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Sex_F",
    "Sex_I",
    "Sex_M"
]

RAW_COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight"
]

CLASS_LABELS = {
    0: "Young Abalone (Age ≤ 8)",
    1: "Medium Age Abalone (Age 9–10)",
    2: "Old Abalone (Age ≥ 11)"
}

CLASS_SHORT = {
    0: "Young",
    1: "Medium",
    2: "Old"
}


# ============================================================
# ADVANCED FNN MODEL STRUCTURE
# ============================================================

class AdvancedFNN(nn.Module):
    def __init__(self, input_dim=10, num_classes=3):
        super(AdvancedFNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    models = {}

    models["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    models["lgbm"] = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    models["cat"] = joblib.load(os.path.join(MODEL_DIR, "cat_model.pkl"))
    models["meta"] = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    models["scaler"] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    fnn = AdvancedFNN(input_dim=10, num_classes=3)
    fnn.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "advanced_fnn.pth"),
            map_location=torch.device("cpu")
        )
    )
    fnn.eval()

    models["fnn"] = fnn

    return models


# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(df, scaler=None):
    df = df.copy()

    rename_map = {
        "Whole weight": "Whole_weight",
        "Shucked weight": "Shucked_weight",
        "Viscera weight": "Viscera_weight",
        "Shell weight": "Shell_weight",
        "WholeWeight": "Whole_weight",
        "ShuckedWeight": "Shucked_weight",
        "VisceraWeight": "Viscera_weight",
        "ShellWeight": "Shell_weight"
    }

    df = df.rename(columns=rename_map)

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.upper().str.strip()
        df["Sex_F"] = (df["Sex"] == "F").astype(int)
        df["Sex_I"] = (df["Sex"] == "I").astype(int)
        df["Sex_M"] = (df["Sex"] == "M").astype(int)
        df = df.drop(columns=["Sex"])

    missing_cols = [col for col in HIDDEN_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected raw columns: {RAW_COLUMNS}"
        )

    df = df[HIDDEN_COLUMNS]
    X = df.astype(float).values

    if scaler is not None:
        X = scaler.transform(X)

    return X.astype(np.float32)


# ============================================================
# STACKING ENSEMBLE PREDICTION
# ============================================================

def predict_stacking(models, X):
    xgb_prob = models["xgb"].predict_proba(X)
    lgbm_prob = models["lgbm"].predict_proba(X)
    cat_prob = models["cat"].predict_proba(X)

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        fnn_prob = torch.softmax(models["fnn"](X_tensor), dim=1).numpy()

    stack_features = np.hstack([
        xgb_prob,
        lgbm_prob,
        cat_prob,
        fnn_prob
    ])

    predictions = models["meta"].predict(stack_features)

    if hasattr(models["meta"], "predict_proba"):
        confidence = models["meta"].predict_proba(stack_features)
    else:
        confidence = None

    model_probs = {
        "XGBoost": xgb_prob,
        "LightGBM": lgbm_prob,
        "CatBoost": cat_prob,
        "Advanced FNN": fnn_prob
    }

    return predictions, confidence, model_probs


# ============================================================
# VISUALISATION HELPERS
# ============================================================

def plot_confidence(confidence_row):
    conf_df = pd.DataFrame({
        "Class": ["Young", "Medium", "Old"],
        "Probability": confidence_row
    })

    fig = px.bar(
        conf_df,
        x="Class",
        y="Probability",
        color="Probability",
        text_auto=".3f",
        template="plotly_dark",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title="Prediction Confidence by Class"
    )

    return fig


def plot_gauge(confidence_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(confidence_value * 100),
        number={"suffix": "%"},
        title={"text": "Final Confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#38bdf8"},
            "steps": [
                {"range": [0, 50], "color": "#334155"},
                {"range": [50, 75], "color": "#1e40af"},
                {"range": [75, 100], "color": "#0284c7"}
            ]
        }
    ))

    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    return fig


def plot_model_probability_comparison(model_probs):
    rows = []

    for model_name, prob in model_probs.items():
        prob_row = prob[0]
        for class_id, value in enumerate(prob_row):
            rows.append({
                "Model": model_name,
                "Class": CLASS_SHORT[class_id],
                "Probability": value
            })

    df_probs = pd.DataFrame(rows)

    fig = px.bar(
        df_probs,
        x="Model",
        y="Probability",
        color="Class",
        barmode="group",
        template="plotly_dark",
        title="Base Model Probability Comparison"
    )

    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    return fig


def plot_feature_profile(input_df):
    value_cols = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight"
    ]

    profile = input_df[value_cols].T.reset_index()
    profile.columns = ["Feature", "Value"]

    fig = px.line(
        profile,
        x="Feature",
        y="Value",
        markers=True,
        template="plotly_dark",
        title="Input Feature Profile"
    )

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    return fig


def create_sample_csv():
    sample = pd.DataFrame({
        "Sex": ["M", "F", "I", "M", "F"],
        "Length": [0.455, 0.530, 0.350, 0.500, 0.615],
        "Diameter": [0.365, 0.420, 0.265, 0.400, 0.480],
        "Height": [0.095, 0.135, 0.090, 0.130, 0.165],
        "Whole weight": [0.5140, 0.6770, 0.2255, 0.6645, 1.1615],
        "Shucked weight": [0.2245, 0.2565, 0.0995, 0.2580, 0.5130],
        "Viscera weight": [0.1010, 0.1415, 0.0485, 0.1330, 0.3010],
        "Shell weight": [0.1500, 0.2100, 0.0700, 0.2400, 0.3050]
    })

    return sample


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🐚 Abalone AI")
st.sidebar.markdown("### Computational Intelligence Dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧠 Models")
st.sidebar.markdown("""
- Baseline FNN
- Advanced FNN
- Fixed Neuro-Fuzzy
- XGBoost
- LightGBM
- CatBoost
- Stacking Ensemble
- SONFIN
""")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 Classes")
st.sidebar.markdown("**Class 0:** Young Abalone")
st.sidebar.markdown("**Class 1:** Medium Age Abalone")
st.sidebar.markdown("**Class 2:** Old Abalone")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🏆 Project Result")
st.sidebar.metric("Best Kaggle Score", "0.808")
st.sidebar.metric("Best Model", "Stacking Ensemble")
st.sidebar.metric("Explainability Model", "SONFIN")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🐚 Abalone Age Classification</div>
    <div class="hero-subtitle">
        A Computational Intelligence dashboard using FNN, Neuro-Fuzzy Systems, SONFIN,
        XGBoost, LightGBM, CatBoost, and Stacking Ensemble learning.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

try:
    models = load_models()
    st.success("✅ Models loaded successfully.")
except Exception as e:
    st.error("❌ Model loading failed. Check that all required model files are inside the models/ folder.")
    st.exception(e)
    st.stop()


# ============================================================
# DASHBOARD METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Best Kaggle Score", "0.808", "Stacking")

with m2:
    st.metric("Total Models", "8", "CI Methods")

with m3:
    st.metric("Input Features", "8", "Abalone Measures")

with m4:
    st.metric("Classes", "3", "Age Groups")


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Manual Prediction",
    "📂 CSV Batch Prediction",
    "📊 Model Insights",
    "📘 Project Info"
])


# ============================================================
# TAB 1: MANUAL PREDICTION
# ============================================================

with tab1:
    st.header("🔮 Manual Abalone Age Prediction")

    st.markdown("""
    <div class="glass-card">
    Enter the physical measurements of an abalone specimen. The stacking ensemble will predict its age group.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.selectbox("Sex", ["M", "F", "I"])
        length = st.number_input("Length", min_value=0.0, max_value=1.0, value=0.455, step=0.001)
        diameter = st.number_input("Diameter", min_value=0.0, max_value=1.0, value=0.365, step=0.001)

    with col2:
        height = st.number_input("Height", min_value=0.0, max_value=2.0, value=0.095, step=0.001)
        whole_weight = st.number_input("Whole weight", min_value=0.0, max_value=5.0, value=0.5140, step=0.0005)
        shucked_weight = st.number_input("Shucked weight", min_value=0.0, max_value=3.0, value=0.2245, step=0.0005)

    with col3:
        viscera_weight = st.number_input("Viscera weight", min_value=0.0, max_value=2.0, value=0.1010, step=0.0005)
        shell_weight = st.number_input("Shell weight", min_value=0.0, max_value=2.0, value=0.1500, step=0.0005)

    input_df = pd.DataFrame({
        "Sex": [sex],
        "Length": [length],
        "Diameter": [diameter],
        "Height": [height],
        "Whole weight": [whole_weight],
        "Shucked weight": [shucked_weight],
        "Viscera weight": [viscera_weight],
        "Shell weight": [shell_weight]
    })

    st.subheader("Input Preview")
    st.dataframe(input_df, use_container_width=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.plotly_chart(plot_feature_profile(input_df), use_container_width=True)

    if st.button("Run AI Prediction", type="primary"):
        try:
            with st.spinner("Running stacking ensemble inference..."):
                X_input = preprocess_input(input_df, scaler=models["scaler"])
                prediction, confidence, model_probs = predict_stacking(models, X_input)

            pred_class = int(prediction[0])

            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-main">Predicted Class {pred_class}</div>
                <div class="prediction-sub">{CLASS_LABELS[pred_class]}</div>
            </div>
            """, unsafe_allow_html=True)

            if confidence is not None:
                final_confidence = np.max(confidence[0])

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.plotly_chart(plot_confidence(confidence[0]), use_container_width=True)

                with chart_col2:
                    st.plotly_chart(plot_gauge(final_confidence), use_container_width=True)

            st.subheader("Base Model Behaviour")
            st.plotly_chart(plot_model_probability_comparison(model_probs), use_container_width=True)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ============================================================
# TAB 2: CSV BATCH PREDICTION
# ============================================================

with tab2:
    st.header("📂 CSV Batch Prediction")

    st.markdown("""
    Upload a CSV file containing the following columns:

    `Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight`
    """)

    sample_df = create_sample_csv()

    st.download_button(
        label="Download Sample CSV Template",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_input.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            with st.spinner("Running batch predictions..."):
                X_batch = preprocess_input(batch_df, scaler=models["scaler"])
                predictions, confidence, _ = predict_stacking(models, X_batch)

            result_df = batch_df.copy()
            result_df["Predicted_Class"] = predictions
            result_df["Prediction_Label"] = result_df["Predicted_Class"].map(CLASS_LABELS)

            if confidence is not None:
                result_df["Confidence_Class_0"] = confidence[:, 0]
                result_df["Confidence_Class_1"] = confidence[:, 1]
                result_df["Confidence_Class_2"] = confidence[:, 2]
                result_df["Max_Confidence"] = confidence.max(axis=1)

            st.subheader("Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            class_counts = result_df["Predicted_Class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]

            fig = px.pie(
                class_counts,
                names="Class",
                values="Count",
                title="Predicted Class Distribution",
                template="plotly_dark",
                hole=0.45
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

            csv_output = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="abalone_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Batch prediction failed. Please check your CSV column names and values.")
            st.exception(e)


# ============================================================
# TAB 3: MODEL INSIGHTS
# ============================================================

with tab3:
    st.header("📊 Model Insights")

    st.markdown("""
    This section summarises the main experimental outcomes from the project.
    """)

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        results_df = pd.DataFrame({
            "Model": [
                "Baseline FNN",
                "Advanced FNN",
                "Fixed Neuro-Fuzzy",
                "XGBoost",
                "LightGBM",
                "CatBoost",
                "Stacking Ensemble",
                "SONFIN"
            ],
            "Testing Accuracy": [
                0.6411,
                0.6567,
                0.6651,
                0.6603,
                0.6483,
                0.6423,
                0.6316,
                0.6447
            ]
        })

        fig = px.bar(
            results_df,
            x="Model",
            y="Testing Accuracy",
            color="Testing Accuracy",
            template="plotly_dark",
            title="Local Testing Accuracy Comparison"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig, use_container_width=True)

    with insight_col2:
        feature_df = pd.DataFrame({
            "Feature": [
                "Shell Weight",
                "Diameter",
                "Whole Weight",
                "Length",
                "Shucked Weight",
                "Height"
            ],
            "Importance": [
                0.29,
                0.22,
                0.18,
                0.15,
                0.10,
                0.06
            ]
        })

        fig2 = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark",
            color="Importance",
            title="Indicative Feature Importance"
        )

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="glass-card">
    <b>Interpretation:</b> The stacking ensemble achieved the strongest hidden-test Kaggle performance,
    while SONFIN provided stronger fuzzy interpretability through adaptive rule construction.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 4: PROJECT INFORMATION
# ============================================================

with tab4:
    st.header("📘 Project Information")

    st.markdown("""
    ### Objective

    This project performs Abalone age classification using Computational Intelligence techniques.

    ### Class Mapping

    - **Class 0:** Age ≤ 8
    - **Class 1:** Age 9–10
    - **Class 2:** Age ≥ 11

    ### Main Methods

    - Feedforward Neural Networks
    - Neuro-Fuzzy Systems
    - Self-Constructing Neuro-Fuzzy Inference System
    - Gradient Boosting
    - Stacking Ensemble Learning

    ### Best Model

    The stacking ensemble achieved the best Kaggle hidden-test score of **0.808**.

    ### Explainability

    SONFIN was used to demonstrate adaptive fuzzy rule generation and interpretable fuzzy reasoning.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Developed for Abalone Age Classification using Computational Intelligence, FNN, Neuro-Fuzzy Systems, SONFIN, and Ensemble Learning."
)
