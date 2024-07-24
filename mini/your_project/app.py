import os
import pickle
import streamlit as st
import pandas as pd

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

# Streamlit app content
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
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    
    # Decode the prediction
    prediction_decoded = le_prognosis.inverse_transform(prediction)
    
    # Display the result
    st.markdown(f'<div class="prediction-result">Predicted Disease: {prediction_decoded[0]}</div>', unsafe_allow_html=True)
