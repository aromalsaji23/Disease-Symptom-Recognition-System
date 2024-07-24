import pandas as pd

# Load the dataset
data = pd.read_csv('medical_data.csv')

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Display the column names
print("Column names in the dataset:")
print(data.columns)
