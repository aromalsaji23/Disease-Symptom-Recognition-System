import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data, fit_scaler=True):
    # Define the columns (replace these with actual columns from your dataset)
    categorical_features = [
        'itching', 'skin_rash', 'nodal_skin_eruptions',
        'continuous_sneezing', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity',
        'ulcers_on_tongue'
    ]
    
    # Define numerical features if you have any (none in this case)
    numerical_features = []  # Update if you have numerical features
    
    # Fill missing values
    data = data.fillna(0)
    
    le = LabelEncoder()
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = le.fit_transform(data[feature].astype(str))
        else:
            print(f"Warning: {feature} is missing from the dataset")

    scaler = StandardScaler()
    
    # Check if numerical features exist in the dataset
    existing_numerical_features = [feature for feature in numerical_features if feature in data.columns]
    
    if fit_scaler and existing_numerical_features:
        data[existing_numerical_features] = scaler.fit_transform(data[existing_numerical_features])
    
    return data, scaler, le

def preprocess_input(input_data, scaler, le):
    # Define columns based on your dataset
    columns = [
        'itching', 'skin_rash', 'nodal_skin_eruptions',
        'continuous_sneezing', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity',
        'ulcers_on_tongue'
    ]
    
    # Convert input_data to DataFrame with appropriate columns
    input_data_df = pd.DataFrame(input_data, columns=columns)
    
    # Encode categorical features
    for feature in columns:
        if feature in input_data_df.columns:
            input_data_df[feature] = le.transform(input_data_df[feature].astype(str))
    
    # Standardize numerical features if any (none in this case)
    
    return input_data_df
