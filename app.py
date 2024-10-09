import streamlit as st
import joblib
import numpy as np
import os

# Load the trained model
model_path = 'bankruptcy_model.pkl'  # Update the path if necessary

# Check if the model file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file '{model_path}' not found. Please ensure the model is saved correctly.")
    st.stop()  # Stop the app if the model cannot be loaded

# Title and description
st.title('Bankruptcy Prediction App')
st.write("""
This app predicts whether a company is at risk of bankruptcy based on various features.
""")

# Collect user input for all features
def get_user_input():
    industrial_risk = st.selectbox('Industrial Risk', [0, 0.5, 1])
    management_risk = st.selectbox('Management Risk', [0, 0.5, 1])
    financial_flexibility = st.selectbox('Financial Flexibility', [0, 0.5, 1])
    credibility = st.selectbox('Credibility', [0, 0.5, 1])
    competitiveness = st.selectbox('Competitiveness', [0, 0.5, 1])
    operating_risk = st.selectbox('Operating Risk', [0, 0.5, 1])
    
    # Store inputs in a dictionary
    user_data = {
        'industrial_risk': industrial_risk,
        'management_risk': management_risk,
        'financial_flexibility': financial_flexibility,
        'credibility': credibility,
        'competitiveness': competitiveness,
        'operating_risk': operating_risk
    }
    
    # Convert to array for prediction
    features = np.array([industrial_risk, management_risk, financial_flexibility, 
                         credibility, competitiveness, operating_risk]).reshape(1, -1)
    
    return features

# Get user input
user_input = get_user_input()

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(user_input)
    result = 'Bankruptcy' if prediction[0] == 1 else 'Non-Bankruptcy'
    st.subheader(f'Prediction: {result}')