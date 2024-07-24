# Save this as check_files.py
import os

for file_name in ['model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'label_encoder_prognosis.pkl']:
    file_path = os.path.join('models', file_name)
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
    else:
        print(f"File found: {file_path}")
