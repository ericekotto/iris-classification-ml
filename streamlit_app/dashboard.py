import os
import streamlit as st
import pandas as pd
import joblib

# --- ZONE DE DEBUG ---
st.write("### 🔍 Diagnostic du serveur")
st.write("Dossier actuel :", os.getcwd())
st.write("Contenu du dossier actuel :", os.listdir("."))

# Tentative de détection du chemin racine
base_path = os.getcwd()
if "streamlit_app" in base_path:
    data_path = os.path.join(base_path, "..", "data", "iris.csv")
    model_path = os.path.join(base_path, "..", "models", "iris_model.joblib")
else:
    data_path = os.path.join(base_path, "data", "iris.csv")
    model_path = os.path.join(base_path, "models", "iris_model.joblib")

st.write(f"Chemin testé pour DATA : {data_path}")
# ----------------------

try:
    df = pd.read_csv(data_path)
    st.success("✅ Données chargées avec succès !")
except Exception as e:
    st.error(f"❌ Erreur DATA : {e}")

try:
    model = joblib.load(model_path)
    st.success("✅ Modèle chargé avec succès !")
except Exception as e:
    st.error(f"❌ Erreur MODÈLE : {e}")