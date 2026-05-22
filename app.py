
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGuard | Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
 
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 
/* ── WHITE background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
.main {
    background: #f5f7fa !important;
    color: #1a202c !important;
    font-family: 'DM Sans', sans-serif !important;
}
 
/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
 
/* ── Main block width ── */
.block-container {
    max-width: 900px !important;
    padding: 2rem 1.5rem 4rem !important;
    margin: 0 auto !important;
}
 
/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(43,108,176,0.10);
    border: 1px solid rgba(43,108,176,0.25);
    color: #2b6cb0;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.4rem, 6vw, 3.8rem) !important;
    font-weight: 800 !important;
    line-height: 1.08 !important;
    color: #1a202c !important;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}
.hero h1 span { color: #2b6cb0; }
.hero p {
    font-size: 1.05rem;
    color: #718096;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
    font-weight: 300;
}
 
/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(43,108,176,0.2), transparent);
    margin: 2rem 0;
}
 
/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2b6cb0;
    margin-bottom: 1rem;
    margin-top: 2rem;
}
 
/* ── Card ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
 
/* ── Widget labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #4a5568 !important;
    margin-bottom: 4px !important;
}
 
/* ── Select box ── */
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e0 !important;
    border-radius: 10px !important;
    color: #1a202c !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #2b6cb0 !important;
    box-shadow: 0 0 0 3px rgba(43,108,176,0.12) !important;
}
 
/* ── Number input — FIX white text ── */
input[type="number"] {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e0 !important;
    border-radius: 10px !important;
    color: #1a202c !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
input[type="number"]:focus {
    border-color: #2b6cb0 !important;
    box-shadow: 0 0 0 3px rgba(43,108,176,0.12) !important;
    outline: none !important;
}
 
/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #2b6cb0 !important;
}
 
/* ── Predict button ── */
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2b6cb0 0%, #4299e1 100%) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.85rem 2rem !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    margin-top: 0.5rem !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 14px rgba(43,108,176,0.3) !important;
}
div.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
 
/* ── Result boxes ── */
.result-safe {
    background: linear-gradient(135deg, #f0fff4, #e6fffa);
    border: 1.5px solid #9ae6b4;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    animation: fadeUp 0.5s ease both;
}
.result-risk {
    background: linear-gradient(135deg, #fff5f5, #fffaf0);
    border: 1.5px solid #feb2b2;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    animation: fadeUp 0.5s ease both;
}
.result-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-safe .result-title { color: #276749; }
.result-risk .result-title { color: #c53030; }
.result-sub {
    font-size: 0.95rem;
    color: #4a5568;
    font-weight: 400;
    line-height: 1.6;
}
 
/* ── Probability meter ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 1.2rem;
}
.prob-label { font-size: 0.8rem; color: #718096; white-space: nowrap; font-weight: 500; }
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #e2e8f0;
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
}
.prob-pct { font-size: 0.9rem; font-weight: 700; white-space: nowrap; min-width: 42px; text-align: right; }
 
/* ── Pills ── */
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1rem; }
.pill {
    background: #ebf8ff;
    border: 1px solid #bee3f8;
    color: #2c5282;
    border-radius: 100px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 500;
}
 
/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    color: #a0aec0;
    font-size: 0.75rem;
    margin-top: 3rem;
    line-height: 1.7;
    font-style: italic;
}
 
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)
 
 
# ── Load model & artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = keras.models.load_model("model.h5")
    with open("scaler.pkl", "rb")  as f: scaler  = pickle.load(f)
    with open("columns.pkl", "rb") as f: columns = pickle.load(f)
    return model, scaler, columns
 
try:
    model, scaler, columns = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error = str(e)
 
 
# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 Deep Learning · ANN · TensorFlow</div>
    <h1>Neuro<span>Guard</span></h1>
    <p>AI-powered stroke risk assessment built on real-world healthcare data.
       Fill in the patient profile below and get an instant prediction.</p>
</div>
""", unsafe_allow_html=True)
 
if not artifacts_ok:
    st.error(f"⚠️ Model files not found. Make sure model.h5, scaler.pkl, and columns.pkl are in the same folder as app.py.\n\nError: {load_error}")
    st.stop()
 
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
 
 
# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Patient Profile</p>', unsafe_allow_html=True)
 
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        age    = st.slider("Age", 1, 100, 55, help="Patient age in years")
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi    = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
 
    with col2:
        avg_glucose   = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 105.0, step=0.1)
        hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
 
    with col3:
        ever_married   = st.selectbox("Ever Married", ["Yes", "No"])
        work_type      = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
 
    smoking_status = st.selectbox(
        "Smoking Status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )
 
    st.markdown('</div>', unsafe_allow_html=True)
 
predict_btn = st.button("⚡  Analyse Stroke Risk", use_container_width=True)
 
 
# ── Prediction ────────────────────────────────────────────────────────────────
THRESHOLD = 0.1
 
if predict_btn:
    raw = {
        "age":               age,
        "avg_glucose_level": avg_glucose,
        "bmi":               bmi,
        "gender":            gender,
        "ever_married":      ever_married,
        "work_type":         work_type,
        "Residence_type":    residence_type,
        "smoking_status":    smoking_status,
        "hypertension":      1 if hypertension == "Yes" else 0,
        "heart_disease":     1 if heart_disease == "Yes" else 0,
    }
 
    df_input  = pd.DataFrame([raw])
    df_encoded = pd.get_dummies(df_input)
    df_final  = df_encoded.reindex(columns=columns, fill_value=0)
    X         = scaler.transform(df_final)
    prob      = float(model.predict(X, verbose=0)[0][0])
    stroke    = prob >= THRESHOLD
 
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)
 
    if stroke:
        st.markdown(f"""
        <div class="result-risk">
            <div class="result-icon">⚠️</div>
            <div class="result-title">Elevated Stroke Risk Detected</div>
            <div class="result-sub">
                The model indicates this patient may be at <strong style="color:#c53030">higher risk</strong>
                of stroke based on the provided profile.<br>
                Please consult a qualified medical professional for a thorough evaluation.
            </div>
            <div class="prob-row">
                <span class="prob-label">Risk Probability</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:linear-gradient(90deg,#e53e3e,#fc8181);"></div>
                </div>
                <span class="prob-pct" style="color:#c53030">{prob*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-icon">✅</div>
            <div class="result-title">Low Stroke Risk</div>
            <div class="result-sub">
                Based on the entered data, the model predicts a <strong style="color:#276749">low risk</strong>
                of stroke for this patient.<br>
                Regular check-ups are still recommended for overall health maintenance.
            </div>
            <div class="prob-row">
                <span class="prob-label">Risk Probability</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:linear-gradient(90deg,#38a169,#68d391);"></div>
                </div>
                <span class="prob-pct" style="color:#276749">{prob*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown(f"""
    <div class="pills">
        <span class="pill">Age: {age}</span>
        <span class="pill">BMI: {bmi}</span>
        <span class="pill">Glucose: {avg_glucose} mg/dL</span>
        <span class="pill">{'Hypertension ✓' if hypertension=='Yes' else 'No Hypertension'}</span>
        <span class="pill">{'Heart Disease ✓' if heart_disease=='Yes' else 'No Heart Disease'}</span>
        <span class="pill">{smoking_status}</span>
        <span class="pill">Threshold: {THRESHOLD}</span>
    </div>
    """, unsafe_allow_html=True)
 
 
# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    NeuroGuard is an academic/research project and is <strong>NOT</strong> a substitute for professional medical advice.<br>
    Always consult a licensed healthcare provider for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)