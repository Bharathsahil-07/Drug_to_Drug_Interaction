"""
Quick Model Evaluation - Get Performance Metrics
"""

import torch
import pandas as pd
import os
import sys
from mt_gat_model import DrugInteractionMTGAT, MTGATTrainer

def evaluate_model():
    print("\n" + "="*70)
    print("DRUG INTERACTION MODEL - PERFORMANCE METRICS")
    print("="*70)
    
    try:
        # Load model checkpoint
        model_path = 'data/trained_model_v2.pt'
        graph_path = 'data/drug_graph_v2.pt'
        
        print(f"\nðŸ“‚ Loading model from: {model_path}")
        checkpoint = torch.load(model_path, weights_only=False)
        
        print(f"ðŸ“‚ Loading graph from: {graph_path}")
        graph_checkpoint = torch.load(graph_path, weights_only=False)
        graph_data = graph_checkpoint['graph_data']
        
        # Model configuration
        input_dim = checkpoint['input_dim']
        config = checkpoint.get('config', {'hidden_dim': 128, 'embedding_dim': 64, 'heads': 4})
        
        print(f"\nðŸ”§ Model Configuration:")
        print(f"   â€¢ Input Dimension: {input_dim}")
        print(f"   â€¢ Hidden Dimension: {config['hidden_dim']}")
        print(f"   â€¢ Embedding Dimension: {config['embedding_dim']}")
        print(f"   â€¢ Attention Heads: {config['heads']}")
        
        # Dataset info
        print(f"\nðŸ“Š Dataset Statistics:")
        print(f"   â€¢ Total Drugs (Nodes): {graph_data.num_nodes:,}")
        print(f"   â€¢ Total Interactions (Edges): {graph_data.edge_index.shape[1] // 2:,}")
        print(f"   â€¢ Feature Dimensions: {graph_data.x.shape[1]}")
        
        # Initialize model
        print(f"\nðŸš€ Initializing model...")
        model = DrugInteractionMTGAT(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            heads=config['heads'],
            dropout=0.2
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print("âœ… Model weights loaded successfully")
        
        # Create trainer
        device = 'cpu'
        print(f"\nðŸ’» Running evaluation on: {device}")
        
        trainer = MTGATTrainer(model, graph_data, device=device)
        
        # Split edges
        print("ðŸ”€ Splitting data (80/10/10)...")
        trainer.split_edges(train_ratio=0.8, val_ratio=0.1)
        
        print(f"\n   â€¢ Train Edges: {trainer.train_edges.shape[1]:,}")
        print(f"   â€¢ Validation Edges: {trainer.val_edges.shape[1]:,}")
        print(f"   â€¢ Test Edges: {trainer.test_edges.shape[1]:,}")
        
        # Evaluate on validation set
        print(f"\n{'='*70}")
        print("ðŸ“ˆ VALIDATION SET METRICS")
        print("="*70)
        print("Evaluating... (this may take 1-2 minutes)")
        
        val_metrics = trainer.evaluate(trainer.val_edges)
        
        print(f"\n{'Metric':<25} {'Value':<15} {'Interpretation'}")
        print("-"*70)
        print(f"{'AUC-ROC':<25} {val_metrics['auc']:<15.4f} {'Discrimination ability'}")
        print(f"{'Accuracy':<25} {val_metrics['accuracy']:<15.4f} {'Overall correctness'}")
        print(f"{'Precision':<25} {val_metrics['precision']:<15.4f} {'Positive predictive value'}")
        print(f"{'Recall':<25} {val_metrics['recall']:<15.4f} {'Sensitivity'}")
        print(f"{'F1-Score':<25} {val_metrics['f1']:<15.4f} {'Balanced performance'}")
        print(f"{'Severity Accuracy':<25} {val_metrics['severity_acc']:<15.4f} {'Risk classification'}")
        
        # Evaluate on test set
        print(f"\n{'='*70}")
        print("ðŸ“Š TEST SET METRICS (FINAL PERFORMANCE)")
        print("="*70)
        print("Evaluating... (this may take 1-2 minutes)")
        
        test_metrics = trainer.evaluate(trainer.test_edges)
        
        print(f"\n{'Metric':<25} {'Value':<15} {'Interpretation'}")
        print("-"*70)
        print(f"{'AUC-ROC':<25} {test_metrics['auc']:<15.4f} {'Discrimination ability'}")
        print(f"{'Accuracy':<25} {test_metrics['accuracy']:<15.4f} {'Overall correctness'}")
        print(f"{'Precision':<25} {test_metrics['precision']:<15.4f} {'Positive predictive value'}")
        print(f"{'Recall':<25} {test_metrics['recall']:<15.4f} {'Sensitivity'}")
        print(f"{'F1-Score':<25} {test_metrics['f1']:<15.4f} {'Balanced performance'}")
        print(f"{'Severity Accuracy':<25} {test_metrics['severity_acc']:<15.4f} {'Risk classification'}")
        
        # Summary for IEEE Paper
        print(f"\n{'='*70}")
        print("ðŸ“ FOR IEEE PAPER - RESULTS SUMMARY")
        print("="*70)
        print(f"""
Test Set Performance:
â”œâ”€â”€ AUC-ROC: {test_metrics['auc']:.4f}
â”œâ”€â”€ Accuracy: {test_metrics['accuracy']:.4f}
â”œâ”€â”€ Precision: {test_metrics['precision']:.4f}
â”œâ”€â”€ Recall: {test_metrics['recall']:.4f}
â”œâ”€â”€ F1-Score: {test_metrics['f1']:.4f}
â””â”€â”€ Severity Accuracy: {test_metrics['severity_acc']:.4f}

Validation Set Performance:
â”œâ”€â”€ AUC-ROC: {val_metrics['auc']:.4f}
â”œâ”€â”€ Accuracy: {val_metrics['accuracy']:.4f}
â”œâ”€â”€ Precision: {val_metrics['precision']:.4f}
â”œâ”€â”€ Recall: {val_metrics['recall']:.4f}
â”œâ”€â”€ F1-Score: {val_metrics['f1']:.4f}
â””â”€â”€ Severity Accuracy: {val_metrics['severity_acc']:.4f}

Dataset:
â”œâ”€â”€ Total Drugs: {graph_data.num_nodes:,}
â”œâ”€â”€ Total Interactions: {graph_data.edge_index.shape[1] // 2:,}
â”œâ”€â”€ Training Edges: {trainer.train_edges.shape[1]:,}
â”œâ”€â”€ Validation Edges: {trainer.val_edges.shape[1]:,}
â””â”€â”€ Test Edges: {trainer.test_edges.shape[1]:,}
        """)
        
        print("="*70)
        print("âœ… Evaluation Complete!")
        print("="*70 + "\n")
        
        return {
            'validation': val_metrics,
            'test': test_metrics
        }
        
    except Exception as e:
        print(f"\nâŒ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    evaluate_model()

