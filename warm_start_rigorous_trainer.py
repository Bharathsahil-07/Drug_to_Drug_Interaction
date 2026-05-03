"""
Drug Interaction Project - Warm-Start (Random Edge) Rigorous Trainer
Disentangles structural vs. feature leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from rigorous_graph_builder import RigorousDrugGraphBuilder
from mt_gat_model import DrugInteractionMTGAT
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.metrics import roc_auc_score

def warm_start_rigorous_train():
    print("="*80)
    print("WARM-START (RANDOM EDGE) RIGOROUS TRAINING")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    drugs_df = pd.read_csv('data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    # Subset to 2000 for speed
    drug_ids = drugs_df['drug_id'].unique()
    np.random.seed(42)
    np.random.shuffle(drug_ids)
    drug_ids = drug_ids[:2000]
    drugs_df = drugs_df[drugs_df['drug_id'].isin(drug_ids)]
    interactions_df = interactions_df[interactions_df['drug_1'].isin(drug_ids) & interactions_df['drug_2'].isin(drug_ids)]
    
    # 2. Split Edges first to identify training drugs
    # (Actually in Warm-start, we allow all drugs in training graph)
    train_indices = list(range(len(drugs_df)))
    
    # 3. Build Graph (Rigorous feature fitting)
    builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
    graph_data = builder.build_graph(train_indices) # Technically fitted on all for warm-start, but rigorous
    
    # 4. Random Edge Split (Warm-start)
    # This is where structural leakage occurs (seen nodes)
    edge_index = graph_data.edge_index
    num_edges = edge_index.shape[1] // 2
    
    perm = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    
    train_pos_idx = perm[:train_size]
    test_pos_idx = perm[train_size:]
    
    # Extract unique (undirected) edges
    edges = edge_index.t()[:num_edges*2:2] # Every second edge
    
    train_edges = edges[train_pos_idx].t()
    test_edges = edges[test_pos_idx].t()
    
    print(f"Edges: {train_edges.shape[1]} train, {test_edges.shape[1]} test (Warm-Start)")
    
    # Undirected training edges for message passing
    train_edges_undirected = torch.cat([train_edges, torch.stack([train_edges[1], train_edges[0]])], dim=1)
    
    # 5. Initialize Model
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionMTGAT(input_dim, hidden_dim=128, embedding_dim=64, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    model.to(device)
    graph_data.to(device)
    train_edges_undirected = train_edges_undirected.to(device)
    
    # 6. Training Loop (Reduced epochs for speed)
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        
        neg_edges = negative_sampling(train_edges_undirected, num_nodes=graph_data.num_nodes, num_neg_samples=train_edges.shape[1])
        edge_label_index = torch.cat([train_edges.to(device), neg_edges.to(device)], dim=1)
        edge_labels = torch.cat([torch.ones(train_edges.shape[1]), torch.zeros(neg_edges.shape[1])]).to(device)
        
        logits, _, _ = model(graph_data.x, train_edges_undirected, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), edge_labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}", flush=True)
            
    # 7. Final Evaluation
    model.eval()
    with torch.no_grad():
        neg_test_edges = negative_sampling(train_edges_undirected, num_nodes=graph_data.num_nodes, num_neg_samples=test_edges.shape[1])
        test_edge_label_index = torch.cat([test_edges.to(device), neg_test_edges.to(device)], dim=1)
        test_labels = torch.cat([torch.ones(test_edges.shape[1]), torch.zeros(neg_test_edges.shape[1])]).cpu().numpy()
        
        logits, _, _ = model(graph_data.x, train_edges_undirected, test_edge_label_index)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        auc = roc_auc_score(test_labels, probs)
        print("\n" + "="*40, flush=True)
        print(f"WARM-START TEST AUC: {auc:.4f}", flush=True)
        print("="*40, flush=True)

if __name__ == "__main__":
    warm_start_rigorous_train()

