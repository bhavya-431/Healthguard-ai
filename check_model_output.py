import sys, os
sys.path.append('src')
from data_preprocessing import load_and_process_data
import torch
import torch.nn.functional as F

data = load_and_process_data('dataset.csv')
labels_inv = {v: k for k, v in data.label_mapping.items()}

from gnn_model import MedicalGNN
model = MedicalGNN(data.num_nodes, 64, data.num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

print("=== Model Output Distribution Check ===")
print(f"Total classes: {data.num_classes}")

# What does the model predict most often for first 20 stored patients?
with torch.no_grad():
    out = model(data)
    probs = F.softmax(out[:20], dim=1)
    preds = probs.argmax(dim=1)
    actual = data.y[:20]

print("\nFirst 20 patient predictions vs actual:")
for i in range(20):
    pred_label = labels_inv[preds[i].item()]
    actual_label = labels_inv[actual[i].item()]
    match = "✓" if preds[i] == actual[i] else "✗"
    print(f"  [{match}] Predicted: {pred_label:<25} | Actual: {actual_label}")
