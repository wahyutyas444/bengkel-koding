import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Kategori Obesitas", layout="centered")
st.title("Prediksi Kategori Obesitas Berdasarkan Data Pengguna")
st.markdown("Silakan isi data berikut untuk melakukan prediksi kategori obesitas.")

# Load model
try:
    model = joblib.load("best_random_forest_model.pkl")
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan file 'best_random_forest_model.pkl' tersedia.")
    st.stop()

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    scaler = None

# Load label encoder
try:
    le = joblib.load("label_encoder.pkl")
    if not isinstance(le, LabelEncoder):
        st.warning("Label encoder yang dimuat bukan LabelEncoder asli. Mapping manual akan digunakan.")
        le = None
except FileNotFoundError:
    le = None

# Form input pengguna
age = st.number_input("Usia", min_value=1, max_value=120, value=25)
height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=65)
fcvc = st.slider("Frekuensi Konsumsi Sayur", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah Makan per Hari", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi Air Harian", 1.0, 3.0, 2.0)
faf = st.slider("Aktivitas Fisik Mingguan", 0.0, 3.0, 1.0)
tue = st.slider("Waktu di Depan Layar", 0.0, 2.0, 1.0)

# Tombol prediksi
if st.button("Prediksi"):
    input_data = np.array([[age, height, weight, fcvc, ncp, ch2o, faf, tue]])

    # Scaling jika scaler tersedia
    if scaler:
        try:
            input_data = scaler.transform(input_data)
        except Exception as e:
            st.error(f"Gagal melakukan scaling: {e}")
            st.stop()

    # Prediksi
    try:
        pred = model.predict(input_data)[0]
        
        # Jika LabelEncoder tersedia dan valid
        if le:
            pred_label = le.inverse_transform([pred])[0]
        else:
            pred_label = str(pred)  # fallback jika tidak bisa di-decode

        st.success(f"Hasil Prediksi: **{pred_label}**")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
