import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('best_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load temp_data_5sector
with open('temp_data_5sector.pkl', 'rb') as f:
    temp_data_5sector = pickle.load(f)

# Variabel Dependenn dan Independen
X = temp_data_5sector[['fire_emission', 'industrial_emission', 'agricultural_emission', 'waste_disposal', 'household_consumption', 'total_emission']]
y = temp_data_5sector['average_temperature']

# Bagi data menjadi data pelatihan dan data pengujian
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model with training data
model.fit(X_train, y_train)

# Judul aplikasi
st.title('Prediksi Kenaikan Suhu Rata-rata')

# Tambahkan input dari pengguna
fire_emission = st.number_input('Emisi dari Kebakaran (kt)', min_value=0.0, help='Emisi yang dihasilkan dari kebakaran')
industrial_emission = st.number_input('Emisi dari Industri (kt)', min_value=0.0, help='Emisi yang dihasilkan dari kegiatan industri')
agricultural_emission = st.number_input('Emisi dari Pertanian (kt)', min_value=0.0, help='Emisi yang dihasilkan dari sektor pertanian')
waste_disposal = st.number_input('Pembuangan Limbah (kt)', min_value=0.0, help='Emisi yang dihasilkan dari pembuangan limbah')
household_consumption = st.number_input('Konsumsi Rumah Tangga (kt)', min_value=0.0, help='Emisi yang dihasilkan dari konsumsi rumah tangga')
total_emission = st.number_input('Total Emisi(kt)', min_value=0.0,)

# Prediksi menggunakan model
input_data = [[fire_emission, industrial_emission, agricultural_emission, waste_disposal, household_consumption, total_emission]]
prediction = model.predict(input_data)

# Tampilkan prediksi
if st.button('Prediksi'):
    prediction = model.predict(input_data)
    st.markdown(f"### Prediksi Kenaikan Suhu: **{prediction[0]}**")



# # Function to predict temperature increase
# def predict_temperature_increase(features):
#     prediction = model.predict(features)
#     return prediction

# # Main function
# def main():
#     st.title('Temperature Increase Prediction App')
#     st.write('This app predicts the temperature increase based on selected features.')

#     # Input features
#     st.sidebar.header('Input Features')
#     fire_emission = st.sidebar.number_input('Fire Emission', value=0)
#     industrial_emission = st.sidebar.number_input('Industrial Emission', value=0)
#     agricultural_emission = st.sidebar.number_input('Agricultural Emission', value=0)
#     waste_disposal = st.sidebar.number_input('Waste Disposal', value=0)
#     household_consumption = st.sidebar.number_input('Household Consumption', value=0)

#     # Predict button
#     if st.sidebar.button('Predict'):
#         # Create feature vector
#         features = pd.DataFrame({
#             'fire_emission': [fire_emission],
#             'industrial_emission': [industrial_emission],
#             'agricultural_emission': [agricultural_emission],
#             'waste_disposal': [waste_disposal],
#             'household_consumption': [household_consumption]
#         })

#         # Predict temperature increase
#         prediction = predict_temperature_increase(features)

#         # Display prediction
#         st.write('Predicted Temperature Increase:', prediction)

# Run the app

