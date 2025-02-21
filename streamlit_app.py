import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Fungsi untuk melatih dan menyimpan model
def train_and_save_model():
    # Dataset Pima Indians Diabetes (bisa menggunakan dataset lain jika diinginkan)
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Membuat model
    model = LogisticRegression()
    model.fit(X, y)

    # Standarisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Menyimpan model dan scaler
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    st.success("Model dan scaler telah disimpan!")

# Memuat model dan scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# UI dengan Streamlit
def main():
    st.title("AI Diagnosa Diabetes")
    st.write("Masukkan data medis untuk mendiagnosis diabetes.")

    # Input dari pengguna
    pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Kadar Glukosa dalam darah', min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input('Tekanan Darah', min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input('Ketebalan Kulit', min_value=0, max_value=100, step=1)
    insulin = st.number_input('Kadar Insulin', min_value=0, max_value=1000, step=1)
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=100.0, step=0.1)
    pedigree_function = st.number_input('Fungsi Keturunan Diabetes', min_value=0.0, max_value=3.0, step=0.1)
    age = st.number_input('Usia', min_value=18, max_value=100, step=1)

    # Buat array dari data input pengguna
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]])

    # Standarisasi data
    user_data_scaled = scaler.transform(user_data)

    # Prediksi menggunakan model
    if st.button('Diagnosa'):
        prediction = model.predict(user_data_scaled)
        if prediction[0] == 1:
            st.write("Prediksi: Pasien berisiko diabetes (1).")
        else:
            st.write("Prediksi: Pasien tidak berisiko diabetes (0).")

if __name__ == '__main__':
    main()

