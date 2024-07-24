import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 500

# Define the columns
columns = [
    'itching', 'skin_rash', 'nodal_skin_eruptions',
    'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'prognosis'
]

# Generate random data
np.random.seed(0)  # For reproducibility
data = {
    'itching': np.random.choice(['Yes', 'No'], num_samples),
    'skin_rash': np.random.choice(['Yes', 'No'], num_samples),
    'nodal_skin_eruptions': np.random.choice(['Yes', 'No'], num_samples),
    'continuous_sneezing': np.random.choice(['Yes', 'No'], num_samples),
    'shivering': np.random.choice(['Yes', 'No'], num_samples),
    'chills': np.random.choice(['Yes', 'No'], num_samples),
    'joint_pain': np.random.choice(['Yes', 'No'], num_samples),
    'stomach_pain': np.random.choice(['Yes', 'No'], num_samples),
    'acidity': np.random.choice(['Yes', 'No'], num_samples),
    'ulcers_on_tongue': np.random.choice(['Yes', 'No'], num_samples),
    'prognosis': np.random.choice(['Disease A', 'Disease B', 'Disease C', 'Disease D'], num_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_filename = 'disease_symptom_dataset.csv'
df.to_csv(csv_filename, index=False)

print(f'Dataset saved as {csv_filename}')
