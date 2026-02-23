import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class MedicalGNN(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_classes, num_layers=2):
        super(MedicalGNN, self).__init__()
        
        # Learnable embeddings for all nodes (Patients + Words)
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # GNN Layers
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = 0.5

    def forward(self, data, x_emb=None):
        x, edge_index = data.x, data.edge_index
        
        if x_emb is None:
            # x is expected to be [num_nodes, 1] containing indices
            x = x.squeeze()
            x = self.embedding(x)
        else:
            x = x_emb
        
        # Message Passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Output
        # We only care about patient nodes for loss, but we return all
        out = self.classifier(x)
        return out
