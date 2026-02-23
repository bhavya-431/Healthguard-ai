import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_preprocessing import load_and_process_data

# Load data multiple times and see if label_mapping is stable
print("=== Checking label_mapping stability ===")
for i in range(3):
    data = load_and_process_data('dataset.csv')
    mapping = data.label_mapping
    labels = list(mapping.keys())
    print(f"Run {i+1}: First 5 labels: {labels[:5]}")
    print(f"         'Psoriasis' -> index {mapping.get('Psoriasis', 'NOT FOUND')}")
    print(f"         'Dengue' -> index {mapping.get('Dengue', 'NOT FOUND')}")
    print()

# Also check if the saved model was trained with the same label ordering
import torch
try:
    saved = torch.load('best_model.pth', map_location='cpu')
    print("Model keys:", list(saved.keys())[:5])
except Exception as e:
    print(f"Could not load model: {e}")
