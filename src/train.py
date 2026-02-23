import torch
import torch.nn as nn
import torch.optim as optim
import os
from gnn_model import MedicalGNN
from data_preprocessing import load_and_process_data

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'dataset.csv')
    
    if not os.path.exists(csv_path):
        print("Dataset not found.")
        return

    print("Processing data...")
    # Run in-memory
    data = load_and_process_data(csv_path)
    
    # Hyperparameters
    HIDDEN_DIM = 64
    LR = 0.01
    EPOCHS = 200 # Fast training for demo
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = data.to(device)
    
    model = MedicalGNN(num_nodes=data.num_nodes, 
                       hidden_dim=HIDDEN_DIM, 
                       num_classes=data.num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        
        # Loss only on training patient nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation / Test
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            # Accuracy on test set
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(current_dir, '..', 'best_model.pth'))
            
    print(f"Training complete. Best Test Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()
