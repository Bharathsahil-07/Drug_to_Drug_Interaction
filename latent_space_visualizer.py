"""
Drug Interaction Project - Latent Space Visualizer
Visualizes GAT drug embeddings using t-SNE to show pharmacological clustering.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from rigorous_graph_builder import RigorousDrugGraphBuilder
from mt_gat_model import DrugInteractionMTGAT
import os

def run_latent_visualization():
    print("="*80)
    print("LATENT SPACE VISUALIZATION (t-SNE)")
    print("="*80)
    
    # 1. Setup Data
    drugs_df = pd.read_csv('data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    drug_ids = drugs_df['drug_id'].unique()
    np.random.seed(42)
    np.random.shuffle(drug_ids)
    drug_subset = drug_ids[:1500]
    
    drugs_df = drugs_df[drugs_df['drug_id'].isin(drug_subset)]
    interactions_df = interactions_df[
        interactions_df['drug_1'].isin(drug_subset) & 
        interactions_df['drug_2'].isin(drug_subset)
    ]
    
    # 2. Build Graph & Model
    builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
    train_indices = list(range(1000))
    graph_data = builder.build_graph(train_indices)
    
    model = DrugInteractionMTGAT(graph_data.x.shape[1], hidden_dim=128, embedding_dim=64, heads=4)
    if os.path.exists('data/rigorous_cold_start_model.pt'):
        model.load_state_dict(torch.load('data/rigorous_cold_start_model.pt', map_location='cpu'))
    model.eval()
    
    # 3. Extract Embeddings & Pharmacological Features
    with torch.no_grad():
        z = model.encode(graph_data.x, graph_data.edge_index).numpy()
        
    # Get pharmacological one-hots (first 9 features of numerical block)
    num_feats, _ = builder.get_raw_features()
    # Assume 9 categories: Analgesics, Antineoplastics, Cardiovascular, etc.
    # We'll assign dominant category for coloring
    primary_category_idx = np.argmax(num_feats[:, :9], axis=1)
    # Mapping indices back to labels (approximate for demo)
    cat_labels = ['Analgesic', 'Antibiotic', 'Antineoplastic', 'Cardio', 'Derm', 'Endo', 'GI', 'Neuro', 'Resp']
    colors = [cat_labels[idx] for idx in primary_category_idx]
    
    # 4. Projection (t-SNE with PCA fallback)
    print("Computing projection...")
    try:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        method = "t-SNE"
        z_2d = proj.fit_transform(z)
    except Exception as e:
        print(f"t-SNE failed ({e}), falling back to PCA...")
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2)
        method = "PCA"
        z_2d = proj.fit_transform(z)
    
    df_plot = pd.DataFrame({
        'x': z_2d[:, 0],
        'y': z_2d[:, 1],
        'Category': colors
    })
    
    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_plot, x='x', y='y', hue='Category', palette='Set1', s=50, alpha=0.7)
    plt.title(f"GAT Drug Embedding Space ({method})", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/latent_space_projection.png', dpi=300)
    plt.close()
    print(f"{method} visualization saved to plots/latent_space_projection.png")

if __name__ == "__main__":
    run_latent_visualization()

