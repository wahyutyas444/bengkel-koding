import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("Prediksi Kategori Obesitas")

st.markdown("Masukkan data berikut untuk memprediksi")

# Load model
try:
    model = joblib.load("best_random_forest_model.pkl")
except:
    st.error("Model tidak ditemukan.")
    st.stop()

# Load scaler jika ada
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

# Load label encoder jika ada
try:
    le = joblib.load("label_encoder.pkl")
except:
    le = None

# Form input pengguna
age = st.number_input("Usia", min_value=1, max_value=120, value=25)
height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=65)
fcvc = st.slider("Frekuensi konsumsi sayur (FCVC)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan per hari (NCP)", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi air harian (CH2O)", 1.0, 3.0, 2.0)
faf = st.slider("Aktivitas fisik mingguan (FAF)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu di depan layar (TUE)", 0.0, 2.0, 1.0)

if st.button("Prediksi"):
    input_data = np.array([[age, height, weight, fcvc, ncp, ch2o, faf, tue]])

    if scaler:
        input_data = scaler.transform(input_data)

    try:
        pred = model.predict(input_data)[0]
        if le:
            pred = le.inverse_transform([pred])[0]
        st.success(f"Hasil Prediksi: **{pred}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
