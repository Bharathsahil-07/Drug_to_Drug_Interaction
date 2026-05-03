"""
Drug Interaction Project - Chemical Diversity Analysis
Analyzes model performance vs. chemical similarity (generalization test).
"""

import torch
import pandas as pd
import numpy as np
import os
from rigorous_graph_builder import RigorousDrugGraphBuilder
from mt_gat_model import DrugInteractionMTGAT
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

def calculate_jaccard(fp1, fp2):
    """Calculate Jaccard similarity between two binary bitsets"""
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    if union == 0: return 0.0
    return intersection / union

def run_diversity_analysis():
    print("="*80)
    print("CHEMICAL DIVERSITY ANALYSIS (Jaccard Similarity on Morgan Fingerprints)")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Data & Splitting
    drugs_df = pd.read_csv('data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    # Subset to 1500 drugs for speed/memory
    drug_ids = drugs_df['drug_id'].unique()
    np.random.seed(42)
    np.random.shuffle(drug_ids)
    drug_subset = drug_ids[:1500]
    
    drugs_df = drugs_df[drugs_df['drug_id'].isin(drug_subset)]
    interactions_df = interactions_df[
        interactions_df['drug_1'].isin(drug_subset) & 
        interactions_df['drug_2'].isin(drug_subset)
    ]
    
    num_test_drugs = int(0.2 * len(drug_subset))
    test_drug_ids = drug_subset[:num_test_drugs]
    train_drug_ids = drug_subset[num_test_drugs:]
    
    drug_to_idx = {drug_id: i for i, drug_id in enumerate(drugs_df['drug_id'].unique())}
    train_indices = [drug_to_idx[did] for did in train_drug_ids]
    
    # 2. Build Rigorous Graph
    builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
    graph_data = builder.build_graph(train_indices)
    
    # Extract Morgan FPs (assume they are at the end of numerical features)
    num_feats, _ = builder.get_raw_features()
    morgan_fps = num_feats[:, -2048:]
    
    # 3. Model & Embedding Pre-computation
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionMTGAT(input_dim, hidden_dim=128, embedding_dim=64, heads=4)
    if os.path.exists('data/rigorous_cold_start_model.pt'):
        model.load_state_dict(torch.load('data/rigorous_cold_start_model.pt', map_location='cpu'))
        print("Loaded saved rigorous model weights.")
    
    model.eval()
    model.to(device)
    graph_data.to(device)
    
    print("Pre-computing drug embeddings...")
    with torch.no_grad():
        # Encode all nodes in one pass
        z = model.encode(graph_data.x, graph_data.edge_index)
        
    idx_to_drug = builder.idx_to_drug
    test_drug_set = set(test_drug_ids)
    train_slice = morgan_fps[train_indices]
    
    # 4. Partition Test Edges into Similarity Bins
    edges = graph_data.edge_index
    num_edges = edges.shape[1] // 2
    
    test_results = []
    print("Analyzing test edges similarity to training set...")
    
    # Track similarity for test drugs once to save time
    test_drug_to_max_sim = {}
    
    for i in range(num_edges):
        u, v = edges[0, 2*i].item(), edges[1, 2*i].item()
        d1, d2 = idx_to_drug[u], idx_to_drug[v]
        
        if d1 in test_drug_set or d2 in test_drug_set:
            # Get prob from pre-computed embeddings
            with torch.no_grad():
                edge_idx = torch.tensor([[u], [v]]).to(device)
                logits = model.decode(z, edge_idx)
                prob = torch.sigmoid(logits).item()
            
            # Calculate Similarity (Sample for speed)
            if len(test_results) < 500: # Limit samples for Demonstration
                if u not in test_drug_to_max_sim:
                    test_drug_to_max_sim[u] = np.max([calculate_jaccard(morgan_fps[u], train_slice[j]) for j in range(len(train_slice))]) if d1 in test_drug_set else 1.0
                if v not in test_drug_to_max_sim:
                    test_drug_to_max_sim[v] = np.max([calculate_jaccard(morgan_fps[v], train_slice[j]) for j in range(len(train_slice))]) if d2 in test_drug_set else 1.0
                
                avg_sim = (test_drug_to_max_sim[u] + test_drug_to_max_sim[v]) / 2.0
                test_results.append((prob, 1, avg_sim))
    
    # 5. Binning
    print("\nGeneralization Results:")
    bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
    for low, high in bins:
        bin_samples = [r for r in test_results if low <= r[2] < high]
        if bin_samples:
            avg_prob = np.mean([r[0] for r in bin_samples])
            print(f" Bin [{low}-{high}]: Count {len(bin_samples)}, Avg Interaction Signal {avg_prob:.4f}")
        else:
            print(f" Bin [{low}-{high}]: No samples.")

if __name__ == "__main__":
    run_diversity_analysis()

