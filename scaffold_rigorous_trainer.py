"""
Drug Interaction Project - Scaffold Split Rigorous Trainer
Tests generalization to novel chemical scaffolds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from rigorous_graph_builder import RigorousDrugGraphBuilder
from mt_gat_model import DrugInteractionMTGAT
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def scaffold_rigorous_train():
    print("="*80)
    print("SCAFFOLD-SPLIT RIGOROUS TRAINING")
    print("="*80)
    
    if not RDKIT_AVAILABLE:
        print("RDKit not found. Cannot perform scaffold split.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    drugs_df = pd.read_csv('data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    # Subset to 2000 for speed
    drug_ids_all = drugs_df['drug_id'].unique()
    np.random.seed(42)
    np.random.shuffle(drug_ids_all)
    drug_ids_all = drug_ids_all[:2000]
    drugs_df = drugs_df[drugs_df['drug_id'].isin(drug_ids_all)]
    
    # 2. Group Drugs by Scaffold
    scaffold_to_drugs = {}
    for _, row in drugs_df.iterrows():
        smiles = str(row.get('smiles', ''))
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            else:
                scaffold = 'none'
        except:
            scaffold = 'none'
            
        if scaffold not in scaffold_to_drugs:
            scaffold_to_drugs[scaffold] = []
        scaffold_to_drugs[scaffold].append(row['drug_id'])
    
    # Sort scaffolds by size (descending preference for balance)
    all_scaffolds = sorted(scaffold_to_drugs.keys(), key=lambda x: len(scaffold_to_drugs[x]), reverse=True)
    
    # Split scaffolds (Inductive split)
    train_drugs = []
    test_drugs = []
    target_test_size = int(0.2 * len(drugs_df))
    
    for scaffold in all_scaffolds:
        if len(test_drugs) < target_test_size:
            test_drugs.extend(scaffold_to_drugs[scaffold])
        else:
            train_drugs.extend(scaffold_to_drugs[scaffold])
            
    print(f"Drugs: {len(train_drugs)} train, {len(test_drugs)} test (Scaffold split)")
    print(f"Unique Scaffolds: {len(all_scaffolds)}")
    
    # 3. Build Graph (Rigorous)
    drug_to_idx = {drug_id: i for i, drug_id in enumerate(drugs_df['drug_id'].unique())}
    train_indices = [drug_to_idx[did] for did in train_drugs]
    
    builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
    graph_data = builder.build_graph(train_indices)
    
    # 4. Filter Edges (Cold-Start Inductive)
    train_drug_set = set(train_drugs)
    test_drug_set = set(test_drugs)
    idx_to_drug = builder.idx_to_drug
    
    edge_index = graph_data.edge_index
    num_edges = edge_index.shape[1] // 2
    
    train_edges = []
    test_edges = []
    for i in range(num_edges):
        u, v = edge_index[0, 2*i].item(), edge_index[1, 2*i].item()
        d1, d2 = idx_to_drug[u], idx_to_drug[v]
        if d1 in train_drug_set and d2 in train_drug_set:
            train_edges.append([u, v])
        elif d1 in test_drug_set or d2 in test_drug_set:
            test_edges.append([u, v])
            
    train_edges = torch.tensor(train_edges).t()
    test_edges = torch.tensor(test_edges).t()
    
    print(f"Edges: {train_edges.shape[1]} train, {test_edges.shape[1]} test (Scaffold split)")
    
    train_edges_undirected = torch.cat([train_edges, torch.stack([train_edges[1], train_edges[0]])], dim=1)
    
    # 5. Initialize Model
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionMTGAT(input_dim, hidden_dim=128, embedding_dim=64, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    model.to(device)
    graph_data.to(device)
    train_edges_undirected = train_edges_undirected.to(device)
    
    # 6. Training Loop
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
            
    # 7. Evaluation
    model.eval()
    with torch.no_grad():
        neg_test_edges = negative_sampling(train_edges_undirected, num_nodes=graph_data.num_nodes, num_neg_samples=test_edges.shape[1])
        test_edge_label_index = torch.cat([test_edges.to(device), neg_test_edges.to(device)], dim=1)
        test_labels = torch.cat([torch.ones(test_edges.shape[1]), torch.zeros(neg_test_edges.shape[1])]).cpu().numpy()
        
        logits, _, _ = model(graph_data.x, train_edges_undirected, test_edge_label_index)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        auc = roc_auc_score(test_labels, probs)
        print("\n" + "="*40, flush=True)
        print(f"SCAFFOLD TEST AUC: {auc:.4f}", flush=True)
        print("="*40, flush=True)

if __name__ == "__main__":
    scaffold_rigorous_train()

