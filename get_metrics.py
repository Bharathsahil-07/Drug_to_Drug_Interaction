"""
Get Model Metrics - Evaluate the trained model and display performance metrics
"""

import torch
import pandas as pd
import os
from mt_gat_model import DrugInteractionMTGAT, MTGATTrainer

def get_model_metrics():
    print("="*70)
    print("LOADING TRAINED MODEL AND EVALUATING METRICS")
    print("="*70)
    
    # Load model checkpoint
    model_path = 'data/trained_model_v2.pt'
    graph_path = 'data/drug_graph_v2.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return
    
    # Load checkpoint
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Load graph data
    print(f"Loading graph from: {graph_path}")
    graph_checkpoint = torch.load(graph_path, weights_only=False)
    graph_data = graph_checkpoint['graph_data']
    
    # Get model config
    input_dim = checkpoint['input_dim']
    config = checkpoint.get('config', {'hidden_dim': 128, 'embedding_dim': 64, 'heads': 4})
    
    print(f"\nModel Configuration:")
    print(f"  Input Dimension: {input_dim}")
    print(f"  Hidden Dimension: {config['hidden_dim']}")
    print(f"  Embedding Dimension: {config['embedding_dim']}")
    print(f"  Attention Heads: {config['heads']}")
    
    # Initialize model
    model = DrugInteractionMTGAT(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        heads=config['heads'],
        dropout=0.2
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nEvaluating on: {device}")
    
    trainer = MTGATTrainer(model, graph_data, device=device)
    
    # Split edges (same as training)
    trainer.split_edges(train_ratio=0.8, val_ratio=0.1)
    
    print("\n" + "="*70)
    print("VALIDATION SET METRICS")
    print("="*70)
    val_metrics = trainer.evaluate(trainer.val_edges)
    
    print(f"  AUC-ROC:           {val_metrics['auc']:.4f}")
    print(f"  Accuracy:          {val_metrics['accuracy']:.4f}")
    print(f"  Precision:         {val_metrics['precision']:.4f}")
    print(f"  Recall:            {val_metrics['recall']:.4f}")
    print(f"  F1 Score:          {val_metrics['f1']:.4f}")
    print(f"  Severity Accuracy: {val_metrics['severity_acc']:.4f}")
    
    print("\n" + "="*70)
    print("TEST SET METRICS (FINAL PERFORMANCE)")
    print("="*70)
    test_metrics = trainer.evaluate(trainer.test_edges)
    
    print(f"  AUC-ROC:           {test_metrics['auc']:.4f}")
    print(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"  Precision:         {test_metrics['precision']:.4f}")
    print(f"  Recall:            {test_metrics['recall']:.4f}")
    print(f"  F1 Score:          {test_metrics['f1']:.4f}")
    print(f"  Severity Accuracy: {test_metrics['severity_acc']:.4f}")
    
    print("\n" + "="*70)
    print("METRICS EXPLANATION")
    print("="*70)
    print("AUC-ROC: Area under ROC curve - ability to distinguish interactions")
    print("         from non-interactions (0.5 = random, 1.0 = perfect)")
    print("\nAccuracy: Correct predictions / total predictions")
    print("          (includes both interactions and non-interactions)")
    print("\nPrecision: True Positives / (True Positives + False Positives)")
    print("           How many predicted interactions are actually real?")
    print("\nRecall: True Positives / (True Positives + False Negatives)")
    print("        How many real interactions did we catch?")
    print("\nF1 Score: Harmonic mean of Precision and Recall")
    print("          Balance between false positives and false negatives")
    print("\nSeverity Accuracy: Correct severity classification for true interactions")
    print("                   (Minor, Moderate, or Major severity levels)")
    print("="*70)
    
    # Dataset statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"  Total Drugs (Nodes):        {graph_data.num_nodes:,}")
    print(f"  Total Interactions (Edges): {graph_data.edge_index.shape[1] // 2:,}")
    print(f"  Feature Dimensions:         {graph_data.x.shape[1]}")
    print(f"  Train Edges:                {trainer.train_edges.shape[1]:,}")
    print(f"  Validation Edges:           {trainer.val_edges.shape[1]:,}")
    print(f"  Test Edges:                 {trainer.test_edges.shape[1]:,}")
    print("="*70)
    
    return {
        'validation': val_metrics,
        'test': test_metrics
    }

if __name__ == "__main__":
    get_model_metrics()

