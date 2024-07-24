import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model, scaler, and label encoders
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('models/label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

with open('models/label_encoder_prognosis.pkl', 'rb') as le_prognosis_file:
    le_prognosis = pickle.load(le_prognosis_file)

# Preprocess input function
def preprocess_input(input_data, scaler, le):
    # Convert input_data to DataFrame with appropriate columns
    columns = [
        'itching', 'skin_rash', 'nodal_skin_eruptions',
        'continuous_sneezing', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity',
        'ulcers_on_tongue'
    ]
    
    input_data_df = pd.DataFrame([input_data], columns=columns)
    
    # Encode categorical features
    for feature in columns:
        if feature in input_data_df.columns:
            input_data_df[feature] = le.transform(input_data_df[feature].astype(str))
    
    # Standardize numerical features if any (none in this case)
    input_data_scaled = scaler.transform(input_data_df)
    
    return input_data_scaled

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Streamlit app title
st.title('Disease Symptom Recognizer')

# Input fields for user
itching = st.selectbox('Itching', ['Yes', 'No'])
skin_rash = st.selectbox('Skin Rash', ['Yes', 'No'])
nodal_skin_eruptions = st.selectbox('Nodal Skin Eruptions', ['Yes', 'No'])
continuous_sneezing = st.selectbox('Continuous Sneezing', ['Yes', 'No'])
shivering = st.selectbox('Shivering', ['Yes', 'No'])
chills = st.selectbox('Chills', ['Yes', 'No'])
joint_pain = st.selectbox('Joint Pain', ['Yes', 'No'])
stomach_pain = st.selectbox('Stomach Pain', ['Yes', 'No'])
acidity = st.selectbox('Acidity', ['Yes', 'No'])
ulcers_on_tongue = st.selectbox('Ulcers on Tongue', ['Yes', 'No'])

# Button to predict
if st.button('Predict'):
    # Convert 'Yes'/'No' to 1/0
    input_data = [
        int(itching == 'Yes'),
        int(skin_rash == 'Yes'),
        int(nodal_skin_eruptions == 'Yes'),
        int(continuous_sneezing == 'Yes'),
        int(shivering == 'Yes'),
        int(chills == 'Yes'),
        int(joint_pain == 'Yes'),
        int(stomach_pain == 'Yes'),
        int(acidity == 'Yes'),
        int(ulcers_on_tongue == 'Yes'),
    ]
    
    # Preprocess the input data
    input_data_processed = preprocess_input(input_data, scaler, le)
    
    # Make a prediction
    prediction = model.predict(input_data_processed)
    
    # Decode the prediction
    prediction_decoded = le_prognosis.inverse_transform(prediction)
    
    # Display the result
    st.markdown(f'<div class="prediction-result">Predicted Disease: {prediction_decoded[0]}</div>', unsafe_allow_html=True)
