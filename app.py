import streamlit as st
import joblib
import numpy as np
import os

# Load the trained model
try:
    model = joblib.load("crop_model.joblib")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Show model type for debugging
st.write(f"Model type: {type(model).__name__}")  # Debug output

# Define or load class names
class_names = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
]

# Check if model has classes_ attribute
if hasattr(model, 'classes_'):
    class_names = model.classes_

# Create the Streamlit app
st.title("Crop Recommendation App")
st.markdown("### Enter soil and climate conditions to get crop recommendations.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=100, value=30)
K = st.number_input("Potassium (K)", min_value=0, max_value=100, value=40)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

if st.button("Predict"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Add scaling if needed (uncomment if you have a scaler)
    # scaler = joblib.load("scaler.joblib")
    # input_data = scaler.transform(input_data)
    
    try:
        prediction = model.predict(input_data)[0]
        try:
            probability = np.max(model.predict_proba(input_data))
        except AttributeError:
            probability = 1.0
            
        st.subheader(f"Recommended Crop: {prediction.capitalize()}")
        st.write(f"Confidence: {probability:.2%}")

        advice = {
            'rice': 'Ensure proper water drainage and maintain consistent moisture.',
            'maize': 'Monitor for pests like corn borer and fungal diseases.',
            # Add more advice...
        }
        st.write("Advice:", advice.get(prediction.lower(), "Regular soil checks and pest monitoring recommended."))
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")