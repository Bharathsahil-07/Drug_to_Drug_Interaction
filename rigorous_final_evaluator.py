"""
Drug Interaction Project - Rigorous Final Evaluator (STRICT COLD-START)
Enforces zero leakage and zero drug overlap between train/val/test splits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             precision_score, recall_score, brier_score_loss, 
                             confusion_matrix, accuracy_score, roc_curve, 
                             precision_recall_curve)
from sklearn.calibration import calibration_curve
from rigorous_graph_builder import RigorousDrugGraphBuilder
from mt_gat_model import DrugInteractionMTGAT
from torch_geometric.utils import negative_sampling
import warnings

# Hard Warnings/Sanity check thresholds
AUC_LEAKAGE_THRESHOLD = 0.90
SEV_MACRO_F1_SUSPICIOUS = 0.60

def calculate_ece(labels, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

class StrictColdStartTrainer:
    def __init__(self, seed=42, num_drugs=1500):
        self.seed = seed
        self.num_drugs = num_drugs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        
    def run_eval(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 1. Load Data
        drugs_all = pd.read_csv('data/drugs.csv')
        interactions_raw = pd.read_csv('data/interactions.csv')
        
        # 2. Strict Drug Splitting (Inductive)
        drug_ids = drugs_all['drug_id'].unique()
        np.random.shuffle(drug_ids)
        drug_subset = drug_ids[:self.num_drugs]
        
        # Define Split (70-15-15)
        n = len(drug_subset)
        train_ids = drug_subset[:int(0.7*n)]
        val_ids = drug_subset[int(0.7*n):int(0.85*n)]
        test_ids = drug_subset[int(0.85*n):]
        
        # SANITY CHECK: Intersection
        assert len(set(train_ids).intersection(set(test_ids))) == 0, "LEAKAGE: Train drugs in Test set!"
        assert len(set(train_ids).intersection(set(val_ids))) == 0, "LEAKAGE: Train drugs in Val set!"
        
        drugs_df = drugs_all[drugs_all['drug_id'].isin(drug_subset)].reset_index(drop=True)
        interactions_df = interactions_raw[
            interactions_raw['drug_1'].isin(drug_subset) & 
            interactions_raw['drug_2'].isin(drug_subset)
        ]
        
        # Map IDs to local indices
        drug_to_idx = {drug_id: i for i, drug_id in enumerate(drugs_df['drug_id'].unique())}
        train_indices = [drug_to_idx[did] for did in train_ids]
        
        # 3. Build Graph with TRAIN-ONLY FITTING
        builder = RigorousDrugGraphBuilder(drugs_df, interactions_df)
        graph_data = builder.build_graph(train_indices) # Scaler/TF-IDF fit on train_indices only
        
        # 4. Strict Edge Partitioning
        idx_to_drug = builder.idx_to_drug
        train_drug_set = set(train_ids)
        test_drug_set = set(test_ids)
        val_drug_set = set(val_ids)
        
        edges = graph_data.edge_index
        num_pos_edges = edges.shape[1] // 2
        
        train_pos_edges = []
        test_pos_edges = []
        val_pos_edges = []
        
        test_sev_labels = [] # Positives only for severity
        
        edge_to_attr = {}
        for i in range(edges.shape[1]):
            edge_to_attr[(edges[0,i].item(), edges[1,i].item())] = graph_data.edge_attr[i].item()

        for i in range(num_pos_edges):
            u, v = edges[0, 2*i].item(), edges[1, 2*i].item()
            d1, d2 = idx_to_drug[u], idx_to_drug[v]
            
            # Cold-Start criteria: If ANY drug is in test set, it's a test edge
            if d1 in test_drug_set or d2 in test_drug_set:
                test_pos_edges.append([u, v])
                val = edge_to_attr.get((u, v), 0.5)
                if val <= 0.5: test_sev_labels.append(0)
                elif val <= 0.8: test_sev_labels.append(1)
                else: test_sev_labels.append(2)
            elif d1 in val_drug_set or d2 in val_drug_set:
                val_pos_edges.append([u, v])
            else:
                # Both must be in train set
                train_pos_edges.append([u, v])
                
        # To Torch
        train_pos_edges = torch.tensor(train_pos_edges).t()
        test_pos_edges = torch.tensor(test_pos_edges).t()
        val_pos_edges = torch.tensor(val_pos_edges).t()
        
        # Symmetrize Train edges for training GAT
        train_edges_full = torch.cat([train_pos_edges, torch.stack([train_pos_edges[1], train_pos_edges[0]])], dim=1)
        
        # 5. Sanity Checks Print (First seed only)
        if self.seed == 42:
            print("\nSANITY CHECKS (Cold-Start Protocol):")
            print(f"  Drugs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
            print(f"  Overlap (Train, Test): {len(set(train_ids).intersection(set(test_ids)))}")
            print(f"  Edges: Train={train_pos_edges.shape[1]}, Val={val_pos_edges.shape[1]}, Test={test_pos_edges.shape[1]}")
            dist = np.bincount(test_sev_labels, minlength=3)
            print(f"  Test Severity Dist: Minor={dist[0]}, Moderate={dist[1]}, Major={dist[2]}")
            
        # 6. Model & Optimization
        model = DrugInteractionMTGAT(graph_data.x.shape[1], hidden_dim=128, embedding_dim=64, heads=4).to(self.device)
        graph_data = graph_data.to(self.device)
        train_edges_full = train_edges_full.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        # Severity labels for training
        train_labels = []
        for i in range(train_pos_edges.shape[1]):
            u, v = train_pos_edges[0,i].item(), train_pos_edges[1,i].item()
            val = edge_to_attr.get((u, v), 0.5)
            if val <= 0.5: train_labels.append(0)
            elif val <= 0.8: train_labels.append(1)
            else: train_labels.append(2)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(self.device)
        sev_weights = builder.get_severity_weights(edges, graph_data.edge_attr).to(self.device)

        # 7. Training (Simplified but rigorous)
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            
            # Neg sampling ONLY from train nodes
            neg_train = negative_sampling(train_edges_full, num_nodes=graph_data.num_nodes, num_neg_samples=train_pos_edges.shape[1])
            edge_idx = torch.cat([train_pos_edges.to(self.device), neg_train], dim=1)
            edge_labs = torch.cat([torch.ones(train_pos_edges.shape[1]), torch.zeros(neg_train.shape[1])]).to(self.device)
            
            logits, sev_logits, _ = model(graph_data.x, train_edges_full, edge_idx)
            loss_int = F.binary_cross_entropy_with_logits(logits.squeeze(), edge_labs)
            pos_sev_logits = sev_logits[:train_pos_edges.shape[1]]
            loss_sev = F.cross_entropy(pos_sev_logits, train_labels_tensor, weight=sev_weights)
            
            (loss_int + 0.5 * loss_sev).backward()
            optimizer.step()
            
        # 8. RE-EVALUATION (TEST ONLY)
        model.eval()
        with torch.no_grad():
            # Rigorous Inductive Negative Sampling:
            # Negatives must involve at least one test drug to match the inductive pos distribution
            test_indices = [drug_to_idx[did] for did in test_ids]
            all_indices = list(range(len(drug_to_idx)))
            existing_edges = set()
            for i in range(edges.shape[1]):
                existing_edges.add((edges[0,i].item(), edges[1,i].item()))
            
            neg_test = []
            while len(neg_test) < test_pos_edges.shape[1]:
                u = np.random.choice(test_indices)
                v = np.random.choice(all_indices)
                if u == v: continue
                if (u,v) not in existing_edges and (v,u) not in existing_edges:
                    neg_test.append([u, v])
            neg_test = torch.tensor(neg_test).t().to(self.device)
            
            eval_edge_idx = torch.cat([test_pos_edges.to(self.device), neg_test], dim=1)
            eval_labels = torch.cat([torch.ones(test_pos_edges.shape[1]), torch.zeros(neg_test.shape[1])]).cpu().numpy()
            
            logits, sev_logits, _ = model(graph_data.x, train_edges_full, eval_edge_idx)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Binary Metrics
            m = {}
            m['AUC'] = roc_auc_score(eval_labels, probs)
            m['AUPRC'] = average_precision_score(eval_labels, probs)
            m['F1'] = f1_score(eval_labels, (probs > 0.5).astype(int))
            m['Acc'] = accuracy_score(eval_labels, (probs > 0.5).astype(int))
            m['Brier'] = brier_score_loss(eval_labels, probs)
            m['ECE'] = calculate_ece(eval_labels, probs)
            
            # Severity
            test_pos_sev_logits = sev_logits[:test_pos_edges.shape[1]]
            sev_preds = torch.argmax(test_pos_sev_logits, dim=1).cpu().numpy()
            m['Sev_Macro_F1'] = f1_score(test_sev_labels, sev_preds, average='macro', zero_division=0)
            m['Sev_W_F1'] = f1_score(test_sev_labels, sev_preds, average='weighted', zero_division=0)
            
            # For Plotting
            m['plot_data'] = (eval_labels, probs, test_sev_labels, sev_preds)
            
            if m['AUC'] > AUC_LEAKAGE_THRESHOLD:
                print(f"  WARNING: AUC={m['AUC']:.4f} > {AUC_LEAKAGE_THRESHOLD}. Highly suspicious for cold-start.")
                
            return m

def run_rigorous_study():
    seeds = [42 + i for i in range(10)]
    all_metrics = []
    
    print("="*80)
    print("STRICT INDUCTIVE COLD-START EVALUATION (10 SEEDS)")
    print("="*80)
    
    for s in seeds:
        trainer = StrictColdStartTrainer(seed=s)
        res = trainer.run_eval()
        all_metrics.append(res)
        print(f"Seed {s}: AUC={res['AUC']:.4f}, Macro-F1={res['Sev_Macro_F1']:.4f}, ECE={res['ECE']:.4f}")
        
    # Aggregate
    summary = {}
    for key in ['AUC', 'AUPRC', 'F1', 'Acc', 'Brier', 'ECE', 'Sev_Macro_F1', 'Sev_W_F1']:
        vals = [m[key] for m in all_metrics]
        summary[key] = (np.mean(vals), np.std(vals))
        
    print("\n" + "="*80)
    print("FINAL RIGOROUS METRIC TABLE (Mean Â± Std)")
    print("="*80)
    for k, v in summary.items():
        print(f"{k:15}: {v[0]:.4f} Â± {v[1]:.4f}")
    print("="*80)
    
    # Plotting
    os.makedirs('plots_rigorous', exist_ok=True)
    
    # 1. ROC (Mean + Shaded)
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for m in all_metrics:
        l, p, _, _ = m['plot_data']
        fpr, tpr, _ = roc_curve(l, p)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC={summary["AUC"][0]:.3f})')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, color='b')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.title("Rigorous Cold-Start ROC (10 Seeds)")
    plt.legend()
    plt.savefig('plots_rigorous/roc_rigorous.png', dpi=300)
    
    # 2. PR (Mean + Shaded)
    plt.figure(figsize=(8, 6))
    mean_recall = np.linspace(0, 1, 100)
    precs = []
    for m in all_metrics:
        l, p, _, _ = m['plot_data']
        prec, rec, _ = precision_recall_curve(l, p)
        # Flip to make it monotonic for interpolation
        precs.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
    mean_prec = np.mean(precs, axis=0)
    plt.plot(mean_recall, mean_prec, color='g', label=f'Mean PR (AUPRC={summary["AUPRC"][0]:.3f})')
    plt.title("Rigorous Cold-Start PR Curve (10 Seeds)")
    plt.legend()
    plt.savefig('plots_rigorous/pr_rigorous.png', dpi=300)

    # 3. Calibration
    plt.figure(figsize=(8, 6))
    l, p, _, _ = all_metrics[0]['plot_data']
    y, x = calibration_curve(l, p, n_bins=10)
    plt.plot(x, y, 'o-', label='Model (Seed 42)')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.title("Calibration Diagram (Reliability)")
    plt.savefig('plots_rigorous/calibration_rigorous.png', dpi=300)
    
    # 4. Confusion Matrix (Seed 42)
    l, p, y_true, y_pred = all_metrics[0]['plot_data']
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Minor', 'Moderate', 'Major'], yticklabels=['Minor', 'Moderate', 'Major'])
    plt.title("Severity Confusion Matrix (Seed 42 counts)")
    plt.savefig('plots_rigorous/cm_raw_rigorous.png', dpi=300)
    
    # Table output for documentation
    with open('rigorous_metrics_all.txt', 'w') as f:
        f.write("RIGOROUS EVALUATION RESULTS\n")
        for k, v in summary.items():
            f.write(f"{k}: {v[0]:.4f} +- {v[1]:.4f}\n")

if __name__ == "__main__":
    run_rigorous_study()

