
import torch
import pandas as pd
import numpy as np
import os
from graph_builder import DrugGraphBuilder
from torch_geometric.data import Data
from mt_gat_model import DrugInteractionMTGAT, MTGATTrainer

def train_new_model():
    print("Initializing training for Upgrade (GAT + Multi-Task)...")
    
    # Load data
    drugs_file = 'data/drugs_enriched.csv' if os.path.exists('data/drugs_enriched.csv') else 'data/drugs.csv'
    interactions_file = 'data/interactions.csv'
    
    print(f"Using drugs file: {drugs_file}")
    
    drugs_df = pd.read_csv(drugs_file)
    interactions_df = pd.read_csv(interactions_file)
    
    if os.path.exists('data/drug_graph_v2.pt'):
        print("Loading existing graph from data/drug_graph_v2.pt...")
        # Load manually to ensure weights_only=False
        checkpoint = torch.load('data/drug_graph_v2.pt', weights_only=False)
        graph_data = checkpoint['graph_data']
        drug_to_idx = checkpoint['drug_to_idx']
        idx_to_drug = checkpoint['idx_to_drug']
        
        # Create a mock builder to hold mappings for saving later
        class MockBuilder:
            pass
        builder = MockBuilder()
        builder.drug_to_idx = drug_to_idx
        builder.idx_to_drug = idx_to_drug
    else:
        # Build Graph
        builder = DrugGraphBuilder(drugs_df, interactions_df)
        # Enable text features and now chemical features will be auto-extracted if columns exist
        graph_data = builder.build_graph(use_text_features=True, max_text_features=50)
        
        # Save the new graph
        builder.save_graph('data/drug_graph_v2.pt')
    
    # Initialize Model
    input_dim = graph_data.x.shape[1]
    # GAT with 4 heads
    model = DrugInteractionMTGAT(input_dim, hidden_dim=128, embedding_dim=64, heads=4, dropout=0.2)
    
    print(f"Model Input Dim: {input_dim}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    trainer = MTGATTrainer(model, graph_data, device=device)
    trainer.split_edges(train_ratio=0.8, val_ratio=0.1)
    
    # Train for enough epochs to converge. 
    # Early stopping handles it.
    metrics = trainer.train(epochs=50, lr=0.001)
    
    # Save
    save_path = 'data/trained_model_v2.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'drug_to_idx': builder.drug_to_idx,
        'idx_to_drug': builder.idx_to_drug,
        'input_dim': input_dim,
        'config': {
            'hidden_dim': 128,
            'embedding_dim': 64,
            'heads': 4
        }
    }, save_path)
    
    print(f"New model saved to {save_path}")

if __name__ == "__main__":
    train_new_model()

