"""
Drug Interaction Project - Cold-Start (Inductive) Trainer
Implements Rigorous Evaluation Protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from rigorous_graph_builder import RigorousDrugGraphBuilder
from mt_gat_model import DrugInteractionMTGAT, MTGATTrainer
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def cold_start_train():
    print("="*80)
    print("COLD-START (INDUCTIVE) MODEL TRAINING")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Data
    drugs_df = pd.read_csv('data/drugs_enriched.csv' if os.path.exists('data/drugs_enriched.csv') else 'data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    # 2. Rigorous Drug-Level Split (Inductive)
    # Subset to 2000 drugs for efficiency on CPU
    drug_ids = drugs_df['drug_id'].unique()
    np.random.seed(42)
    np.random.shuffle(drug_ids)
    drug_ids = drug_ids[:2000] 
    
    num_test_drugs = int(0.2 * len(drug_ids))
    test_drug_ids = drug_ids[:num_test_drugs]
    train_drug_ids = drug_ids[num_test_drugs:]
    
    # Indices for RigorousDrugGraphBuilder
    drug_to_idx = {drug_id: i for i, drug_id in enumerate(drugs_df['drug_id'].unique())}
    train_indices = [drug_to_idx[did] for did in train_drug_ids]
    
    print(f"Drugs: {len(train_drug_ids)} train, {len(test_drug_ids)} test (Cold-Start)")
    
    # 3. Build Graph (Rigorous fitting)
    builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
    graph_data = builder.build_graph(train_indices)
    
    # 4. Filter Edges for Cold-Start
    # Training edges: only interactions between train_drug_ids
    # Test edges: interactions involving at least one test_drug_id
    
    train_drug_set = set(train_drug_ids)
    test_drug_set = set(test_drug_ids)
    
    edge_index = graph_data.edge_index
    num_edges = edge_index.shape[1] // 2
    
    train_edges = []
    test_edges = []
    
    # Map back to IDs
    idx_to_drug = builder.idx_to_drug
    
    for i in range(num_edges):
        u, v = edge_index[0, 2*i].item(), edge_index[1, 2*i].item()
        d1, d2 = idx_to_drug[u], idx_to_drug[v]
        
        if d1 in train_drug_set and d2 in train_drug_set:
            train_edges.append([u, v])
        elif d1 in test_drug_set or d2 in test_drug_set:
            test_edges.append([u, v])
            
    train_edges = torch.tensor(train_edges).t()
    test_edges = torch.tensor(test_edges).t()
    
    print(f"Edges: {train_edges.shape[1]} train, {test_edges.shape[1]} test (Cold-Start)")
    
    # Undirected training edges for message passing
    train_edges_undirected = torch.cat([train_edges, torch.stack([train_edges[1], train_edges[0]])], dim=1)
    
    # 5. Severity Weights & Labels (Optimized)
    edge_to_attr = {}
    full_edges = graph_data.edge_index
    full_attrs = graph_data.edge_attr
    for i in range(full_edges.shape[1]):
        edge_to_attr[(full_edges[0,i].item(), full_edges[1,i].item())] = full_attrs[i].item()
    
    train_labels = []
    for i in range(train_edges.shape[1]):
        u, v = train_edges[0,i].item(), train_edges[1,i].item()
        val = edge_to_attr.get((u, v), 0.5)
        if val <= 0.5: train_labels.append(0)
        elif val <= 0.8: train_labels.append(1)
        else: train_labels.append(2)
    
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
    sev_weights = builder.get_severity_weights(edge_index, graph_data.edge_attr).to(device)
    print(f"Severity Class Weights: {sev_weights.tolist()}")
    
    # 6. Initialize Model & Optimizer
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionMTGAT(input_dim, hidden_dim=128, embedding_dim=64, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    model.to(device)
    graph_data.to(device)
    train_edges_undirected = train_edges_undirected.to(device)
    
    # 7. Training Loop
    best_auc = 0
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        # Negative sampling on training graph
        neg_edges = negative_sampling(train_edges_undirected, num_nodes=graph_data.num_nodes, num_neg_samples=train_edges.shape[1])
        
        edge_label_index = torch.cat([train_edges.to(device), neg_edges.to(device)], dim=1)
        edge_labels = torch.cat([torch.ones(train_edges.shape[1]), torch.zeros(neg_edges.shape[1])]).to(device)
        
        logits, sev_logits, conf = model(graph_data.x, train_edges_undirected, edge_label_index)
        
        # Interaction Loss
        loss_int = F.binary_cross_entropy_with_logits(logits.squeeze(), edge_labels)
        
        # Severity Loss (Weighted) - only for the positive training edges
        loss_sev = torch.tensor(0.0).to(device)
        if train_edges.shape[1] > 0:
            pos_sev_logits = sev_logits[:train_edges.shape[1]]
            loss_sev = F.cross_entropy(pos_sev_logits, train_labels_tensor, weight=sev_weights)
        
        loss = loss_int + 0.5 * loss_sev
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}", flush=True)
            
    # 8. Final Evaluation (Cold-Start)
    model.eval()
    with torch.no_grad():
        neg_test_edges = negative_sampling(train_edges_undirected, num_nodes=graph_data.num_nodes, num_neg_samples=test_edges.shape[1])
        test_edge_label_index = torch.cat([test_edges.to(device), neg_test_edges.to(device)], dim=1)
        test_labels = torch.cat([torch.ones(test_edges.shape[1]), torch.zeros(neg_test_edges.shape[1])]).cpu().numpy()
        
        logits, _, _ = model(graph_data.x, train_edges_undirected, test_edge_label_index)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        auc = roc_auc_score(test_labels, probs)
        print("\n" + "="*40, flush=True)
        print(f"COLD-START TEST AUC: {auc:.4f}", flush=True)
        print("="*40, flush=True)
        
    # Save model
    torch.save(model.state_dict(), 'data/rigorous_cold_start_model.pt')
    print("\nâœ… Rigorous Model Saved to data/rigorous_cold_start_model.pt")

if __name__ == "__main__":
    cold_start_train()

