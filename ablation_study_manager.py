"""
Drug Interaction Project - Ablation Study Manager
Compares Multi-Task vs. Single-Task performance under rigorous inductive splits.
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, precision_score, recall_score
from scipy import stats

def calculate_ece(labels, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if sample list in bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

class AblationTrainer:
    def __init__(self, mode='multi_task', num_seeds=5, num_drugs=2000):
        self.mode = mode
        self.num_seeds = num_seeds
        self.num_drugs = num_drugs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def run_single_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 1. Load & Split
        drugs_df = pd.read_csv('data/drugs.csv')
        interactions_df = pd.read_csv('data/interactions.csv')
        
        drug_ids_all = drugs_df['drug_id'].unique()
        np.random.shuffle(drug_ids_all)
        drug_subset = drug_ids_all[:self.num_drugs]
        
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
        
        # 2. Build Graph
        builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
        graph_data = builder.build_graph(train_indices)
        
        # 3. Induction Split
        idx_to_drug = builder.idx_to_drug
        train_drug_set = set(train_drug_ids)
        test_drug_set = set(test_drug_ids)
        
        edges = graph_data.edge_index
        num_edges = edges.shape[1] // 2
        
        train_edges = []
        test_edges = []
        full_test_sev_labels = [] # To calculate Severity Macro-F1
        
        # Pre-map labels for train/test edges
        edge_to_attr = {}
        for i in range(graph_data.edge_index.shape[1]):
            u, v = graph_data.edge_index[0,i].item(), graph_data.edge_index[1,i].item()
            edge_to_attr[(u, v)] = graph_data.edge_attr[i].item()

        for i in range(num_edges):
            u, v = edges[0, 2*i].item(), edges[1, 2*i].item()
            d1, d2 = idx_to_drug[u], idx_to_drug[v]
            if d1 in train_drug_set and d2 in train_drug_set:
                train_edges.append([u, v])
            elif d1 in test_drug_set or d2 in test_drug_set:
                test_edges.append([u, v])
                val = edge_to_attr.get((u, v), 0.5)
                if val <= 0.5: full_test_sev_labels.append(0)
                elif val <= 0.8: full_test_sev_labels.append(1)
                else: full_test_sev_labels.append(2)
                
        train_edges = torch.tensor(train_edges).t()
        test_edges = torch.tensor(test_edges).t()
        train_edges_undirected = torch.cat([train_edges, torch.stack([train_edges[1], train_edges[0]])], dim=1)
        
        # Severity weights
        sev_weights = builder.get_severity_weights(edges, graph_data.edge_attr).to(self.device)
        
        train_labels = []
        for i in range(train_edges.shape[1]):
            u, v = train_edges[0,i].item(), train_edges[1,i].item()
            val = edge_to_attr.get((u, v), 0.5)
            if val <= 0.5: train_labels.append(0)
            elif val <= 0.8: train_labels.append(1)
            else: train_labels.append(2)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(self.device)

        # 4. Model
        input_dim = graph_data.x.shape[1]
        model = DrugInteractionMTGAT(input_dim, hidden_dim=128, embedding_dim=64, heads=4)
        model.to(self.device)
        graph_data.to(self.device)
        train_edges_undirected = train_edges_undirected.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        # 5. Training
        for epoch in range(1, 51):
            model.train()
            optimizer.zero_grad()
            neg_edges = negative_sampling(train_edges_undirected, num_nodes=graph_data.num_nodes, num_neg_samples=train_edges.shape[1])
            edge_label_index = torch.cat([train_edges.to(self.device), neg_edges.to(self.device)], dim=1)
            edge_labels = torch.cat([torch.ones(train_edges.shape[1]), torch.zeros(neg_edges.shape[1])]).to(self.device)
            
            logits, sev_logits, _ = model(graph_data.x, train_edges_undirected, edge_label_index)
            loss_int = F.binary_cross_entropy_with_logits(logits.squeeze(), edge_labels)
            
            if self.mode == 'multi_task':
                pos_sev_logits = sev_logits[:train_edges.shape[1]]
                loss_sev = F.cross_entropy(pos_sev_logits, train_labels_tensor, weight=sev_weights)
                loss = loss_int + 0.5 * loss_sev
            else:
                loss = loss_int
            loss.backward()
            optimizer.step()
            
        # 6. Comprehensive Evaluation
        model.eval()
        metrics = {}
        with torch.no_grad():
            neg_test_edges = negative_sampling(train_edges_undirected, num_nodes=graph_data.num_nodes, num_neg_samples=test_edges.shape[1])
            test_edge_label_index = torch.cat([test_edges.to(self.device), neg_test_edges.to(self.device)], dim=1)
            test_labels = torch.cat([torch.ones(test_edges.shape[1]), torch.zeros(neg_test_edges.shape[1])]).cpu().numpy()
            
            logits, sev_logits, _ = model(graph_data.x, train_edges_undirected, test_edge_label_index)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Binary Metrics
            metrics['AUC'] = roc_auc_score(test_labels, probs)
            metrics['AUPRC'] = average_precision_score(test_labels, probs)
            metrics['F1'] = f1_score(test_labels, (probs > 0.5).astype(int))
            metrics['Precision'] = precision_score(test_labels, (probs > 0.5).astype(int))
            metrics['Recall'] = recall_score(test_labels, (probs > 0.5).astype(int))
            metrics['Brier'] = brier_score_loss(test_labels, probs)
            metrics['ECE'] = calculate_ece(test_labels, probs)
            
            # Severity Metrics (only if MT)
            if self.mode == 'multi_task':
                test_pos_sev_logits = sev_logits[:test_edges.shape[1]]
                sev_preds = torch.argmax(test_pos_sev_logits, dim=1).cpu().numpy()
                metrics['Sev_Macro_F1'] = f1_score(full_test_sev_labels, sev_preds, average='macro')
            else:
                metrics['Sev_Macro_F1'] = 0.0
                
            return metrics

def run_ablation_study():
    num_seeds = 10
    num_drugs = 1500
    
    print("="*80)
    print(f"ABLATION STUDY: Comprehensive Metrics (10 Seeds)")
    print("="*80)
    
    results = {'multi_task': [], 'single_task': []}
    
    for i in range(num_seeds):
        seed = 42 + i
        print(f"\nSeed {i+1}/{num_seeds} (Seed: {seed})")
        
        for mode in ['multi_task', 'single_task']:
            trainer = AblationTrainer(mode=mode, num_seeds=num_seeds, num_drugs=num_drugs)
            m = trainer.run_single_seed(seed)
            results[mode].append(m)
            print(f"  {mode.upper():12} -> AUC: {m['AUC']:.4f}, AUPRC: {m['AUPRC']:.4f}, ECE: {m['ECE']:.4f}")
            
    print("\n" + "="*80)
    print("SUMMARY PERFORMANCE (Publication-Ready)")
    print("="*80)
    
    for mode in ['multi_task', 'single_task']:
        print(f"\n--- {mode.upper()} ---")
        for metric in ['AUC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Brier', 'ECE', 'Sev_Macro_F1']:
            vals = [r[metric] for r in results[mode]]
            print(f"{metric:15}: {np.mean(vals):.4f} Â± {np.std(vals):.4f}")
            
    # T-test for AUPRC and AUC
    t_auc, p_auc = stats.ttest_rel([r['AUC'] for r in results['multi_task']], [r['AUC'] for r in results['single_task']])
    t_prc, p_prc = stats.ttest_rel([r['AUPRC'] for r in results['multi_task']], [r['AUPRC'] for r in results['single_task']])
    
    print("\nStatistical Significance (MT vs ST):")
    print(f"  AUC   P-value: {p_auc:.6f}")
    print(f"  AUPRC P-value: {p_prc:.6f}")
    print("="*80)

if __name__ == "__main__":
    run_ablation_study()

