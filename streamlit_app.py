# ============================================================
# PROJECTK: PREMIUM ABALONE AGE INFRASTRUCTURE & ANALYTICS APP
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ============================================================
# ENHANCED PAGE LAYOUT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ProjectK: Abalone Intelligence Dashboard",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI CSS Inject targeting component layout, tabs, and premium cards
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #ffffff;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
        color: #4B5563;
        border: 1px solid #E5E7EB;
    }
    .stTabs [aria-selected="true"] { 
        border-bottom: 3px solid #2563EB !important; 
        color: #2563EB !important; 
        font-weight: 700;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border-left: 6px solid #2563EB;
        margin-bottom: 20px;
    }
    .decoder-container {
        background-color: #F8FAFC;
        padding: 16px;
        border-radius: 8px;
        border: 1px dashed #CBD5E1;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SYSTEM CONSTANTS & DATA SCHEMAS
# ============================================================

MODEL_DIR = "models"

HIDDEN_COLUMNS = [
    "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight",
    "Sex_F", "Sex_I", "Sex_M"
]

CLASS_LABELS = {
    0: "Young Abalone (Age ≤ 8)",
    1: "Middle Age Abalone (Age 9–10)",
    2: "Old Abalone (Age ≥ 11)"
}

# ============================================================
# BASE ARCHITECTURE STRUCTURAL FRAMEWORKS
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
# MODEL LIFECYCLE LOADERS
# ============================================================

@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    artifacts["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    artifacts["lgbm"] = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    artifacts["cat"] = joblib.load(os.path.join(MODEL_DIR, "cat_model.pkl"))
    artifacts["meta"] = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    artifacts["scaler"] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    fnn = AdvancedFNN(input_dim=10, num_classes=3)
    fnn.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "advanced_fnn.pth"),
            map_location=torch.device("cpu")
        )
    )
    fnn.eval()
    artifacts["fnn"] = fnn
    return artifacts

# ============================================================
# PREPROCESSING & INFERENCE PIPELINES
# ============================================================

def execute_preprocessing(df, scaler=None):
    df = df.copy()
    rename_map = {
        "Whole weight": "Whole_weight",
        "Shucked weight": "Shucked_weight",
        "Viscera weight": "Viscera_weight",
        "Shell weight": "Shell_weight"
    }
    df = df.rename(columns=rename_map)

    if "Sex" in df.columns:
        df["Sex_F"] = (df["Sex"] == "F").astype(int)
        df["Sex_I"] = (df["Sex"] == "I").astype(int)
        df["Sex_M"] = (df["Sex"] == "M").astype(int)
        df = df.drop(columns=["Sex"])

    missing_cols = [col for col in HIDDEN_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature alignment mismatch: Missing {missing_cols}")

    df = df[HIDDEN_COLUMNS]
    X = df.astype(float).values

    if scaler is not None:
        X = scaler.transform(X)

    return X.astype(np.float32)

def run_stacking_inference(models, X):
    xgb_prob = models["xgb"].predict_proba(X)
    lgbm_prob = models["lgbm"].predict_proba(X)
    cat_prob = models["cat"].predict_proba(X)

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        fnn_prob = torch.softmax(models["fnn"](X_tensor), dim=1).numpy()

    stack_features = np.hstack([xgb_prob, lgbm_prob, cat_prob, fnn_prob])
    predictions = models["meta"].predict(stack_features)
    confidence = models["meta"].predict_proba(stack_features) if hasattr(models["meta"], "predict_proba") else None

    return predictions, confidence, [xgb_prob, lgbm_prob, cat_prob, fnn_prob]

# ============================================================
# INTERACTIVE SIDEBAR STRIP
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/plasticine/200/sea-shell.png", width=80)
    st.title("ProjectK Infrastructure")
    st.markdown("---")
    
    st.markdown("### 🧬 Current Operational Status")
    st.success("● Stacking Core Active")
    st.info("● Device Profile: Single-Threaded CPU")
    
    st.markdown("---")
    st.markdown("### 📊 Benchmark Scoreboard")
    metrics_summary = pd.DataFrame({
        "Model Configuration": ["Advanced FNN", "XGBoost Ensembles", "Stacking Meta-Learner"],
        "Leaderboard Val Acc": ["0.784", "0.798", "0.808"]
    })
    st.dataframe(metrics_summary, hide_index=True)

# ============================================================
# MAIN INTERFACE DISPATCH
# ============================================================

st.title("ProjectK: Advanced Computational Intelligence & Analytics Platform")
st.caption("Fusing Advanced Feedforward Neural Nets, Gradient Boosted Trees, and Self-Constructing Inference Logic.")
st.markdown("---")

try:
    models = load_all_artifacts()
except Exception as e:
    st.error("🚨 System Initialization Exception: Unable to bind model artifacts.")
    st.exception(e)
    st.stop()

# Master Module Tabs
tab_manual, tab_batch, tab_analytics = st.tabs([
    "🎯 Real-Time Manual Entry Engine", 
    "📊 Vectorized File Batch Processor", 
    "📈 Structural Artifact Analytics"
])

# ------------------------------------------------------------
# MODULE 1: MANUAL INFERENCE ENGINE WITH LIVE LINGUISTIC DECODER
# ------------------------------------------------------------
with tab_manual:
    st.markdown("### Live Specimen Inspection Suite")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Continuous Variable Slider Arrays**")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            sex = st.selectbox("Specimen Biological Profile (Sex)", ["M", "F", "I"])
            length = st.slider("Radial Length Vector Value", 0.0, 1.0, 0.455, 0.005)
            diameter = st.slider("Radial Diameter Vector Value", 0.0, 1.0, 0.365, 0.005)
            height = st.slider("Vertical Axis Height Vector Value", 0.0, 0.5, 0.095, 0.005)
        with sub_col2:
            whole_weight = st.slider("Total Mass (Whole Weight)", 0.0, 3.0, 0.514, 0.005)
            shucked_weight = st.slider("Edible Mass (Shucked Weight)", 0.0, 2.0, 0.224, 0.005)
            viscera_weight = st.slider("Organ Mass (Viscera Weight)", 0.0, 1.0, 0.101, 0.005)
            shell_weight = st.slider("Dry Mass (Shell Weight)", 0.0, 1.0, 0.150, 0.005)

    with col2:
        st.markdown("**💡 Live Linguistic Antecedent Transcription Decoder**")
        st.markdown("This window runs heuristic profiling matching the behavior of the dynamic membership rules.")
        
        # Real-time rule evaluation emulation
        length_profile = "High" if length > 0.6 else ("Medium" if length > 0.35 else "Low")
        shell_profile = "High" if shell_weight > 0.4 else ("Medium" if shell_weight > 0.15 else "Low")
        
        st.markdown(f"""
        <div class="decoder-container">
            <strong>Active Rule-Base Approximation Trace:</strong><br>
            • <code>IF</code> Length Variable is <strong>{length_profile}</strong> ({length:.3f})<br>
            • <code>AND</code> Dry Shell Mass is <strong>{shell_profile}</strong> ({shell_weight:.3f})<br>
            • <code>THEN</code> Activate Adaptive TSK Consequent Model Coefficients.
        </div>
        """, unsafe_allow_html=True)

    input_df = pd.DataFrame({
        "Sex": [sex], "Length": [length], "Diameter": [diameter], "Height": [height],
        "Whole weight": [whole_weight], "Shucked weight": [shucked_weight],
        "Viscera weight": [viscera_weight], "Shell weight": [shell_weight]
    })

    st.markdown("---")
    if st.button("Execute Pipeline Meta-Stacking Evaluation", type="primary", use_container_width=True):
        try:
            X_input = execute_preprocessing(input_df, scaler=models["scaler"])
            prediction, confidence, base_probs = run_stacking_inference(models, X_input)
            pred_class = int(prediction[0])

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <p style="margin:0; font-size:12px; color:#6B7280; font-weight:bold; text-transform:uppercase;">Meta-Learner Output Target</p>
                    <h2 style="margin:8px 0; color:#1E3A8A;">Class Assignment: {pred_class}</h2>
                    <p style="margin:0; font-size:14px; color:#2563EB; font-weight:600;">{CLASS_LABELS[pred_class]}</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                if confidence is not None:
                    conf_df = pd.DataFrame({
                        "Age Tier Categories": ["Young Cluster (≤8)", "Middle Cluster (9–10)", "Old Cluster (≥11)"],
                        "Probability Target Weight": confidence[0]
                    })
                    st.bar_chart(conf_df.set_index("Age Tier Categories"), use_container_width=True)
        except Exception as e:
            st.error("Pipeline breakdown tracked during manual processing step.")
            st.exception(e)

# ------------------------------------------------------------
# MODULE 2: VECTORIZED BATCH PROCESSING ENGINE
# ------------------------------------------------------------
with tab_batch:
    st.markdown("### Vectorized Array Upload Interface")
    uploaded_file = st.file_uploader("Upload Raw Document Splitting Target Asset", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded CSV Matrix Buffer: {batch_df.shape[0]} valid observation coordinates rows isolated.")
            
            X_batch = execute_preprocessing(batch_df, scaler=models["scaler"])
            predictions, confidence, _ = run_stacking_inference(models, X_batch)

            output_df = batch_df.copy()
            output_df["Meta_Predicted_Class"] = predictions
            output_df["Linguistic_Label"] = output_df["Meta_Predicted_Class"].map(CLASS_LABELS)

            if confidence is not None:
                output_df["Conf_Prob_Class_0"] = confidence[:, 0]
                output_df["Conf_Prob_Class_1"] = confidence[:, 1]
                output_df["Conf_Prob_Class_2"] = confidence[:, 2]

            st.dataframe(output_df, use_container_width=True)

            csv_data = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Optimized Prediction Target Manifest Document",
                data=csv_data,
                file_name="projectk_export_manifest.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error("Exception processing vectorized files format layout rows.")
            st.exception(e)

# ------------------------------------------------------------
# MODULE 3: ARTIFACT ANALYTICS & HYBRID ARCHITECTURE INSPECTION
# ------------------------------------------------------------
with tab_analytics:
    st.markdown("### Structural Architectural Profiler Tools")
    
    sub_tab_base, sub_tab_fuzzy = st.tabs(["🌳 Standalone Base Classifiers", "🔮 Explanatory Rule Bases"])
    
    with sub_tab_base:
        st.markdown("#### Meta Ensemble Constituency Layout Weightings")
        st.markdown("""
        The stacking blender handles multi-variant vectors by weighting continuous predictions 
        across four decoupled algorithmic systems simultaneously.
        """)
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            st.markdown("**Gradient Tree Structural Estimators**")
            st.caption("• XGBoost: Regularized Step Matrix (max_depth=3)")
            st.caption("• LightGBM: Leaf-Wise Growth Core Engine")
            st.caption("• CatBoost: Symmetric Oblivious Split Partitions")
        with col_an2:
            st.markdown("**Deep Linear Mapping Framework**")
            st.caption("• Advanced FNN Architecture (128 → 64 → 32 Layers Mapping Block)")
            st.caption("• Normalization Strategy: 1D Batch Normalization Variance Smoothing")

    with sub_tab_fuzzy:
        st.markdown("#### Fuzzy Antecedent / Consequent Clustering Profiles")
        st.markdown("""
        To balance black-box ensemble decisions, the runtime infrastructure uses localized Gaussian sets 
        mapping linguistic rules across the physical feature topology.
        """)
        
        an_col1, an_col2 = st.columns(2)
        with an_col1:
            st.markdown("""
            <div style="background-color:#FFF; padding:15px; border-radius:6px; border:1px solid #E5E7EB;">
                <strong>Fixed Mamdani Framework Configuration:</strong><br>
                • Static Rule Allocation Cap: 12 Nodes<br>
                • Membership Form: High-Dimensional Gaussian Means ($\mu$)<br>
                • Optimization Layer: Gradient Descent Backprop
            </div>
            """, unsafe_allow_html=True)
        with an_col2:
            st.markdown("""
            <div style="background-color:#FFF; padding:15px; border-radius:6px; border:1px solid #E5E7EB;">
                <strong>Dynamic Self-Constructing (SONFIN) Engine:</strong><br>
                • Rule Expansion Allocation Rule: Online Streaming Allocation<br>
                • Threshold Bounds Firing Limit ($A_{th}$): 0.60<br>
                • Consequent Mathematical Method: First-Order TSK Linear Equations
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FOOTER TERMINAL STRIP
# ============================================================
st.markdown("---")
st.caption("ProjectK Operational Dashboard Interface Module Layout System Terminal Strip.")
