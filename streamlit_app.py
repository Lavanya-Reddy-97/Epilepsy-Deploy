import streamlit as st
import numpy as np
import pickle

# Load the trained model and scalers
model_path = "model/Epilepsy_model.pkl"
scaler_X_path = "model/scaler.pkl"
scaler_y_path = "model/scaler_y.pkl"

# Load model and scalers
final_model = pickle.load(open(model_path, 'rb'))
scaler_X = pickle.load(open(scaler_X_path, 'rb'))
scaler_y = pickle.load(open(scaler_y_path, 'rb'))

# Streamlit App Title
st.title("🧠 Epilepsy Risk Prediction App")

st.write("""
### Enter the following medical parameters to predict the risk percentage:
""")

# User Input Fields
mean = st.number_input("Mean Value", format="%.6f")
variance = st.number_input("Variance", format="%.6f")
std_dev = st.number_input("Standard Deviation", format="%.6f")
skewness = st.number_input("Skewness", format="%.6f")
kurtosis = st.number_input("Kurtosis", format="%.6f")
entropy = st.number_input("Entropy", format="%.6f")
fft_mean = st.number_input("FFT Mean", format="%.6f")
fft_max = st.number_input("FFT Max", format="%.6f")
delta_power = st.number_input("Delta Power", format="%.6f")
theta_power = st.number_input("Theta Power", format="%.6f")
alpha_power = st.number_input("Alpha Power", format="%.6f")
beta_power = st.number_input("Beta Power", format="%.6f")
gamma_power = st.number_input("Gamma Power", format="%.6f")

# Make Prediction on Button Click
if st.button("Predict Epilepsy's Risk"):
    try:
        # Arrange input data
        input_data = np.array([[mean, variance, std_dev, skewness, kurtosis, entropy, fft_mean, fft_max, delta_power, theta_power, alpha_power, beta_power, gamma_power]])

        # Apply feature scaling
        input_data_scaled = scaler_X.transform(input_data)

        # Make Prediction
        pred_scaled = final_model.predict(input_data_scaled)

        # Convert prediction back to original scale
        risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

        # Risk Status Classification
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

        # Display Results
        st.success(f"🧠 **Predicted Epilepsy's Risk Percentage:** {risk_percent:.2f}%")
        st.info(f"**Risk Status:** {risk_status}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
