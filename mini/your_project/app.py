import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import base64

# Determine the absolute path
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'models', 'model.pkl')
scaler_path = os.path.join(base_path, 'models', 'scaler.pkl')
le_path = os.path.join(base_path, 'models', 'label_encoder.pkl')
le_prognosis_path = os.path.join(base_path, 'models', 'label_encoder_prognosis.pkl')

# Load the model, scaler, and label encoders
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(le_path, 'rb') as le_file:
    le = pickle.load(le_file)

with open(le_prognosis_path, 'rb') as le_prognosis_file:
    le_prognosis = pickle.load(le_prognosis_file)

# CSS styles with the new background image settings
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://wallpaperaccess.com/full/3512568.jpg");
            background-attachment: fixed;
            background-size: cover;
        }
        .stTitle {
            color: black;  /* Set the title heading color to black */
        }
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            color: #000000; /* Green color for the result */
            margin-top: 20px;
        }
        .prediction-probability {
            font-size: 20px;
            color: #000000; /* Red color for the probability */
            margin-top: 10px;
        }
        .dataframe {
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
        }
        h3 {
            color: #000000; /* Blue color for the headings */
        }
        .container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            border-radius: 8px;
            background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app content
st.markdown('<div class="container">', unsafe_allow_html=True)
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
    
    input_data_df = pd.DataFrame([input_data], columns=[
        'itching', 'skin_rash', 'nodal_skin_eruptions',
        'continuous_sneezing', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity',
        'ulcers_on_tongue'
    ])
    
    # Preprocess the input data
    input_data_scaled = scaler.transform(input_data_df)
    
    # Make a prediction and get probabilities
    probabilities = model.predict_proba(input_data_scaled)
    
    # Decode the prediction
    prediction_decoded = le_prognosis.inverse_transform([np.argmax(probabilities)])
    
    # Get the probability for the predicted disease
    predicted_index = np.argmax(probabilities)
    predicted_probability = probabilities[0][predicted_index]

    # Display the result
    st.markdown(f'<div class="prediction-result">Predicted Disease: {prediction_decoded[0]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-probability">Chance of having {prediction_decoded[0]}: {predicted_probability * 100:.2f}%</div>', unsafe_allow_html=True)

    # Show all probabilities for each disease
    disease_names = le_prognosis.inverse_transform(range(len(probabilities[0])))
    probabilities_df = pd.DataFrame({
        'Disease': disease_names,
        'Probability': probabilities[0] * 100
    })
    probabilities_df = probabilities_df.sort_values(by='Probability', ascending=False)
    
    # Display the probabilities
    st.write("### Chances of Each Disease:")
    st.dataframe(probabilities_df)

st.markdown('</div>', unsafe_allow_html=True)
