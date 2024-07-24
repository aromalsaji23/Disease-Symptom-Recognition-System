import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load your dataset
data = pd.read_csv('data/medical_data.csv')

# Define the columns
columns = [
    'itching', 'skin_rash', 'nodal_skin_eruptions',
    'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'prognosis'
]

# Select relevant columns
data = data[columns]

# Encode categorical features
le = LabelEncoder()
for feature in columns[:-1]:  # Skip the target column 'prognosis'
    data[feature] = le.fit_transform(data[feature].astype(str))

# Split data into features and target
X = data[columns[:-1]]
y = le.fit_transform(data['prognosis'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features if any (none in this case)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model, scaler, and label encoder
with open('models/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('models/label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

print("Model, scaler, and label encoder saved successfully!")
