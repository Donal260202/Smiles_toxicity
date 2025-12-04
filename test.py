import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from mordred import Calculator, descriptors
from tensorflow.keras.models import load_model
from io import BytesIO

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="SMILES Toxicity Predictor", layout="wide")

# ===============================
# CONFIG
# ===============================
MODEL_FILES = [
    "ensemble_model_1.h5",
    "ensemble_model_2.h5",
    "ensemble_model_3.h5",
    "ensemble_model_4.h5",
    "ensemble_model_5.h5"
]

X_SCALER_FILE = "x_scaler.pkl"
Y_SCALER_FILE = "y_scaler.pkl"
FEATURE_FILE = "selected_mordred_features.txt"

# ===============================
# CACHE: LOAD MODELS & SCALERS
# ===============================
@st.cache_resource
def load_all_models():
    models = [load_model(m, compile=False) for m in MODEL_FILES]
    x_scaler = joblib.load(X_SCALER_FILE)
    y_scaler = joblib.load(Y_SCALER_FILE)

    with open(FEATURE_FILE) as f:
        features = [line.strip() for line in f if line.strip() != ""]

    return models, x_scaler, y_scaler, features

models, x_scaler, y_scaler, REQ_FEATURES = load_all_models()

# ===============================
# MORDRED CALCULATOR
# ===============================
calc = Calculator(descriptors, ignore_3D=True)

def compute_mordred(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return calc(mol)
    except:
        return None

# ===============================
# DOWNLOAD BUTTON
# ===============================
def download_excel(df, filename):
    output = BytesIO()
    df.to_excel(output, index=False)
    st.download_button(
        "⬇️ Download Predictions",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===============================
# UI
# ===============================
st.title("🧪 SMILES-Based Toxicity Predictor (5-Model Ensemble)")
st.markdown("Upload an Excel file with a column named **`smiles`**.")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

# ===============================
# PROCESS
# ===============================
if uploaded_file:

    df = pd.read_excel(uploaded_file)

    if "smiles" not in df.columns:
        st.error("❌ Column 'smiles' not found!")
        st.stop()

    with st.spinner("🔬 Computing Mordred descriptors..."):
        desc_series = df["smiles"].apply(compute_mordred)
        df_desc = pd.DataFrame(desc_series.tolist())

    # ===============================
    # SELECT & CLEAN FEATURES
    # ===============================
    df_selected = df_desc.reindex(columns=REQ_FEATURES)  # Keep only required features
    df_selected = df_selected.replace([np.inf, -np.inf], np.nan)

    # Fill missing Mordred features with column mean
    df_selected = df_selected.fillna(df_selected.mean())

    # Warn about molecules that were completely invalid
    valid_idx = df_selected.dropna(how='all').index
    removed = len(df) - len(valid_idx)
    if removed > 0:
        st.warning(f"⚠️ {removed} molecules removed due to completely invalid descriptors.")

    df = df.loc[valid_idx].reset_index(drop=True)
    df_selected = df_selected.loc[valid_idx].reset_index(drop=True)

    if df.empty:
        st.error("❌ No valid SMILES remaining after cleaning.")
        st.stop()

    # ===============================
    # SCALE
    # ===============================
    X_scaled = x_scaler.transform(df_selected)

    # ===============================
    # ENSEMBLE PREDICTION
    # ===============================
    with st.spinner("🤖 Running ensemble prediction..."):
        all_preds = []
        for model in models:
            p = model.predict(X_scaled, verbose=0).ravel()
            all_preds.append(p)
        mean_pred_scaled = np.mean(all_preds, axis=0)

    # ===============================
    # INVERSE SCALE
    # ===============================
    y_pred = y_scaler.inverse_transform(mean_pred_scaled.reshape(-1, 1)).ravel()

    # ===============================
    # OUTPUT
    # ===============================
    df["Toxicity_Prediction"] = y_pred.round(4)

    # ===============================
    # DISPLAY
    # ===============================
    st.success("✅ Prediction Completed!")

    st.subheader("📊 Results (Top 20)")
    st.dataframe(df.head(20), use_container_width=True)

    # ===============================
    # DOWNLOAD
    # ===============================
    st.subheader("⬇️ Download Results")
    download_excel(df, "toxicity_predictions.xlsx")
