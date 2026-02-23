import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
import re
import os
from collections import Counter

def clean_text(text):
    text = str(text).lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

def load_and_process_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Encode labels
    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label_idx'] = df['label'].map(label_mapping)
    num_classes = len(label_mapping)
    print(f"Found {num_classes} unique diseases.")
    
    # Process text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Build Vocabulary (Symptom Nodes)
    all_words = []
    for text in df['clean_text']:
        all_words.extend(text.split())
    
    # Filter common stopwords (basic list to avoid NLTK dependency issues if not installed)
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
    
    word_counts = Counter(all_words)
    # k-core filtering or frequency filtering? Let's verify commonly used words but keep meaningful ones
    # For medical context, we want to keep most specific words.
    
    vocab = [w for w, c in word_counts.items() if w not in stop_words and len(w) > 2]
    vocab = sorted(list(set(vocab)))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Construct Graph
    # Patient Nodes: indices 0 to len(df)-1
    # Word Nodes: indices len(df) to len(df)+len(vocab)-1
    
    num_patients = len(df)
    num_words = len(vocab)
    
    # Edges: Patient <-> Word
    src_nodes = []
    dst_nodes = []
    
    print("Building graph edges...")
    for patient_idx, row in df.iterrows():
        words = row['clean_text'].split()
        for w in set(words): # edges are unique per patient-word pair
            if w in word_to_idx:
                word_node_idx = num_patients + word_to_idx[w]
                
                # Undirected graph for GCN
                # Patient to Word
                src_nodes.append(patient_idx)
                dst_nodes.append(word_node_idx)
                
                # Word to Patient
                src_nodes.append(word_node_idx)
                dst_nodes.append(patient_idx)
    
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    
    # Node Features
    # We can use One-Hot or simple Identity for now, but embedding layer is better handled in model
    # Let's create a feature matrix `x`.
    # For heterogeneous-like split in a homogeneous graph:
    # We will let the model learn embeddings.
    # x stores node indices to be passed to an Embedding layer.
    
    # Wait, 'x' usually features. If we want to learn embeddings, we pass scalar indices.
    # However, GCN inputs `x` as [num_nodes, num_features].
    # Let's start with Identity matrix? No, too big.
    # Let's assume the model will have an `Embedding` layer that takes node IDs.
    # But usually GNN take `x`.
    # Let's define `x` as a dummy constant feature for now, or better:
    # Use TF-IDF for patients?
    # Simple approach: x is just a feature placeholder, we will use `node_id` in the model if needed, 
    # OR we can treat it as featureless nodes and assign random embeddings.
    # Let's assign an integer ID as feature, so the model can embed it.
    
    x = torch.arange(num_patients + num_words, dtype=torch.long).unsqueeze(1)
    
    # Labels
    # Only patient nodes have labels.
    y = torch.tensor(df['label_idx'].values, dtype=torch.long)
    
    # Masks
    # Random 80/20 split
    perm = torch.randperm(num_patients)
    train_size = int(0.8 * num_patients)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    
    train_mask = torch.zeros(num_patients + num_words, dtype=torch.bool)
    test_mask = torch.zeros(num_patients + num_words, dtype=torch.bool)
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # We extend classes to align with y shape potentially, but `y` passed to loss usually matches mask
    # For safety, let's pad y with -1 for word nodes, though we won't use them in loss
    y_extended = torch.full((num_patients + num_words,), -1, dtype=torch.long)
    y_extended[:num_patients] = y
    
    data = Data(x=x, edge_index=edge_index, y=y_extended)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.num_patients = num_patients
    data.num_words = num_words
    data.num_classes = num_classes
    data.label_mapping = label_mapping
    data.vocab = word_to_idx
    
    print("Graph construction complete.")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    
    return data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming dataset is in the parent directory or known path
    # user workspace: c:/Users/bhavy/OneDrive/Desktop/antigravity gnn/dataset.csv
    csv_path = os.path.join(current_dir, '..', 'dataset.csv')
    
    if os.path.exists(csv_path):
        data = load_and_process_data(csv_path)
        save_path = os.path.join(current_dir, '..', 'processed_data.pt')
        torch.save(data, save_path)
        print(f"Data saved to {save_path}")
    else:
        print(f"Error: {csv_path} not found.")

