"""
Drug Interaction Project - Publication Visualization Suite
Generates high-impact figures for manuscript submission.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import os

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

class PublicationPlotter:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def plot_roc_curves(self, results_dict):
        """
        results_dict: { 'Model Name': (labels, probs) }
        """
        plt.figure(figsize=(8, 6))
        for name, (labels, probs) in results_dict.items():
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Inductive DDI Prediction (Cold-Start)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/roc_curves.png', dpi=300)
        plt.close()

    def plot_pr_curves(self, results_dict):
        plt.figure(figsize=(8, 6))
        for name, (labels, probs) in results_dict.items():
            precision, recall, _ = precision_recall_curve(labels, probs)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (AUPRC = {pr_auc:.3f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pr_curves.png', dpi=300)
        plt.close()

    def plot_calibration(self, results_dict):
        plt.figure(figsize=(8, 6))
        for name, (labels, probs) in results_dict.items():
            prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', lw=2, label=name)
            
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Reliability Diagram (Calibration)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/calibration.png', dpi=300)
        plt.close()

    def plot_diversity_generalization(self, bins, values):
        """
        bins: ['0-0.3', '0.3-0.6', '0.6-1.0']
        values: AUC values
        """
        plt.figure(figsize=(8, 5))
        sns.barplot(x=bins, y=values, palette='viridis')
        plt.ylim([0.5, 1.0])
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Baseline')
        plt.xlabel('Max Jaccard Similarity to Training Set')
        plt.ylabel('AUC Performance')
        plt.title('Performance vs. Chemical Diversity Gap')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/diversity_analysis.png', dpi=300)
        plt.close()

# Example Usage (Placeholder data for demonstration)
if __name__ == "__main__":
    plotter = PublicationPlotter()
    
    # Mock data based on our actual session results
    labels = np.random.randint(0, 2, 1000)
    # GAT (Better)
    probs_gat = (labels * 0.8) + (np.random.rand(1000) * 0.4)
    probs_gat = np.clip(probs_gat, 0, 1)
    # GCN (Baseline)
    probs_gcn = (labels * 0.7) + (np.random.rand(1000) * 0.5)
    probs_gcn = np.clip(probs_gcn, 0, 1)
    
    data_dict = {
        'MT-GAT (Proposed)': (labels, probs_gat),
        'GCN (Baseline)': (labels, probs_gcn)
    }
    
    plotter.plot_roc_curves(data_dict)
    plotter.plot_pr_curves(data_dict)
    plotter.plot_calibration(data_dict)
    
    plotter.plot_diversity_generalization(['0-0.3 (Novel)', '0.3-0.6 (Mid)', '0.6-1.0 (Close)'], [0.782, 0.814, 0.841])
    
    print("Publication figures generated in 'plots/' directory.")
