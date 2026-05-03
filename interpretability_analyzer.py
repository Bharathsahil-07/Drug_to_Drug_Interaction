"""
Drug Interaction Project - GAT Attention Analyzer
Extracts and visualizes attention weights for clinical interpretability.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rigorous_graph_builder import RigorousDrugGraphBuilder
from mt_gat_model import DrugInteractionMTGAT
import os

def visualize_attention_hub(drug_name, neighborhood_df, attention_weights, output_path):
    """
    Visualizes a drug hub where edge thickness corresponds to GAT attention.
    """
    G = nx.Graph()
    G.add_node(drug_name, type='center')
    
    # Sort by attention and take top neighbors
    neighborhood_df['attention'] = attention_weights
    top_neighbors = neighborhood_df.sort_values('attention', ascending=False).head(10)
    
    for _, row in top_neighbors.iterrows():
        G.add_node(row['neighbor'], type='neighbor')
        G.add_edge(drug_name, row['neighbor'], weight=row['attention'] * 5)
        
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, 
                           node_color=['#ff9999' if G.nodes[n]['type'] == 'center' else '#99ccff' for n in G],
                           alpha=0.8)
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.5)
    
    plt.title(f"GAT Local Attention Hub: {drug_name}", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Graph saved to {output_path}")

def run_interpretability_analysis():
    print("="*80)
    print("GAT ATTENTION INTERPRETABILITY ANALYSIS")
    print("="*80)
    
    # 1. Load Data
    drugs_df = pd.read_csv('data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    # Use same subset as cold-start for consistency
    drug_ids = drugs_df['drug_id'].unique()
    np.random.seed(42)
    np.random.shuffle(drug_ids)
    drug_subset = drug_ids[:1500]
    
    drugs_df = drugs_df[drugs_df['drug_id'].isin(drug_subset)]
    interactions_df = interactions_df[
        interactions_df['drug_1'].isin(drug_subset) & 
        interactions_df['drug_2'].isin(drug_subset)
    ]
    
    # 2. Build Graph
    builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
    train_indices = list(range(1000)) # Placeholder split logic consistent with prev
    graph_data = builder.build_graph(train_indices)
    
    # 3. Model
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionMTGAT(input_dim, hidden_dim=128, embedding_dim=64, heads=4)
    if os.path.exists('data/rigorous_cold_start_model.pt'):
        model.load_state_dict(torch.load('data/rigorous_cold_start_model.pt', map_location='cpu'))
        print("Loaded saved rigorous model weights.")
    model.eval()
    
    # 4. Attention Extraction
    # Target: Aspirin (DB00945) or Warfarin (DB00682) if in subset
    target_drug = 'Aspirin' 
    if target_drug in builder.drug_to_idx:
        target_idx = builder.drug_to_idx[target_drug]
        neighbors = graph_data.edge_index[1, graph_data.edge_index[0] == target_idx].tolist()
        
        # Get attention weights from Layer 3 (multi-head)
        with torch.no_grad():
            z, (att_edge_index, att_weights) = model.encode(graph_data.x, graph_data.edge_index, return_attention=True)
            
        # Filter weights for target drug edges
        mask = (att_edge_index[0] == target_idx)
        target_att_weights = att_weights[mask].mean(dim=1).numpy() # Mean across heads
        target_neighbors_idx = att_edge_index[1][mask].tolist()
        
        neighbor_names = [builder.idx_to_drug[idx] for idx in target_neighbors_idx]
        
        neigh_df = pd.DataFrame({'neighbor': neighbor_names})
        
        os.makedirs('plots', exist_ok=True)
        visualize_attention_hub(target_drug, neigh_df, target_att_weights, 'plots/aspirin_attention_hub.png')
    else:
        print(f"{target_drug} not in the current data subset. Searching for common drug...")
        # Fallback to the drug with most neighbors in the subset
        counts = torch.bincount(graph_data.edge_index[0])
        max_idx = torch.argmax(counts).item()
        target_drug_name = builder.idx_to_drug[max_idx]
        
        mask = (graph_data.edge_index[0] == max_idx)
        neighbor_indices = graph_data.edge_index[1][mask].tolist()
        neighbor_names = [builder.idx_to_drug[idx] for idx in neighbor_indices]
        
        with torch.no_grad():
            _, (att_edge_index, att_weights) = model.encode(graph_data.x, graph_data.edge_index, return_attention=True)
            
        mask_att = (att_edge_index[0] == max_idx)
        target_att_weights = att_weights[mask_att].mean(dim=1).numpy()
        
        neigh_df = pd.DataFrame({'neighbor': [builder.idx_to_drug[i] for i in att_edge_index[1][mask_att].tolist()]})
        visualize_attention_hub(target_drug_name, neigh_df, target_att_weights, f'plots/{target_drug_name.lower()}_attention_hub.png')

if __name__ == "__main__":
    run_interpretability_analysis()

