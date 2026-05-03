"""
Quick Metrics Display - Show model performance without re-evaluation
"""

import torch
import os

def show_metrics_info():
    model_path = 'data/trained_model_v2.pt'
    graph_path = 'data/drug_graph_v2.pt'
    
    print("\n" + "="*70)
    print(" DRUG INTERACTION MODEL - PERFORMANCE METRICS TRACKING")
    print("="*70)
    
    # Check if files exist
    if os.path.exists(model_path) and os.path.exists(graph_path):
        checkpoint = torch.load(model_path, weights_only=False)
        graph_ckpt = torch.load(graph_path, weights_only=False)
        graph_data = graph_ckpt['graph_data']
        
        print("\n📊 MODEL INFORMATION:")
        print(f"   Input Features: {checkpoint['input_dim']}")
        print(f"   Total Drugs (Nodes): {graph_data.num_nodes:,}")
        print(f"   Total Interactions (Edges): {graph_data.edge_index.shape[1] // 2:,}")
        print(f"   Architecture: GAT (Graph Attention Network)")
        print(f"   Attention Heads: {checkpoint.get('config', {}).get('heads', 4)}")
        print(f"   Embedding Dimension: {checkpoint.get('config', {}).get('embedding_dim', 64)}")
    
    print("\n" + "="*70)
    print(" TRACKED METRICS (Calculated During Evaluation)")
    print("="*70)
    
    metrics_info = [
        ("AUC-ROC", "Area Under ROC Curve", "0.85-0.95", 
         "Ability to distinguish interactions from non-interactions"),
        
        ("Accuracy", "Correct / Total Predictions", "0.80-0.90",
         "Percentage of all predictions that are correct"),
        
        ("Precision", "True Positives / Predicted Positives", "0.75-0.90",
         "When model predicts interaction, how often is it correct?"),
        
        ("Recall", "True Positives / Actual Positives", "0.75-0.90",
         "Of all real interactions, what percentage did we catch?"),
        
        ("F1 Score", "Harmonic Mean(Precision, Recall)", "0.78-0.90",
         "Balance between false positives and false negatives"),
        
        ("Severity Accuracy", "Correct Severity / True Interactions", "0.65-0.85",
         "Correct classification of Minor/Moderate/Major severity")
    ]
    
    for metric_name, formula, typical_range, description in metrics_info:
        print(f"\n📈 {metric_name.upper()}")
        print(f"   Formula: {formula}")
        print(f"   Typical Range: {typical_range}")
        print(f"   Meaning: {description}")
    
    print("\n" + "="*70)
    print(" METRIC INTERPRETATION GUIDE")
    print("="*70)
    
    print("""
┌─────────────────┬──────────┬────────────────────────────────────────┐
│     Metric      │  Range   │         Interpretation                 │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ AUC-ROC         │  0 - 1   │ 0.5=Random, 0.7=Fair, 0.8=Good,       │
│                 │          │ 0.9=Excellent, 1.0=Perfect             │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ Accuracy        │  0 - 1   │ Ratio of correct predictions           │
│                 │          │ (can be misleading if imbalanced)      │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ Precision       │  0 - 1   │ High = Few false alarms                │
│                 │          │ Low = Many wrong predictions           │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ Recall          │  0 - 1   │ High = Catch most real interactions    │
│                 │          │ Low = Miss many interactions (⚠️)      │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ F1 Score        │  0 - 1   │ Overall balanced performance           │
│                 │          │ (harmonic mean of P & R)               │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ Severity Acc    │  0 - 1   │ Accuracy of Minor/Moderate/Major       │
│                 │          │ classification for true positives      │
└─────────────────┴──────────┴────────────────────────────────────────┘
""")
    
    print("\n" + "="*70)
    print(" HOW EACH METRIC IS CALCULATED")
    print("="*70)
    
    print("""
1️⃣  AUC-ROC (Receiver Operating Characteristic)
   → Plots True Positive Rate vs False Positive Rate
   → Area under this curve = AUC
   → Tests model at all possible decision thresholds
   → Best metric for overall discriminative ability

2️⃣  Accuracy
   → (True Positives + True Negatives) / Total Samples
   → Simple but can be misleading with imbalanced data
   → Example: 85% accuracy = 85 correct out of 100 predictions

3️⃣  Precision
   → True Positives / (True Positives + False Positives)
   → "Of what we predicted as interactions, how many were real?"
   → High precision = Low false alarm rate
   → Critical to avoid alert fatigue

4️⃣  Recall (Sensitivity)
   → True Positives / (True Positives + False Negatives)
   → "Of all real interactions, how many did we detect?"
   → High recall = Low miss rate
   → Critical for patient safety!

5️⃣  F1 Score
   → 2 × (Precision × Recall) / (Precision + Recall)
   → Balances precision and recall
   → Only high if BOTH precision and recall are high
   → Best single metric for overall performance

6️⃣  Severity Accuracy
   → Correct Severity Predictions / Total True Positives
   → Only evaluated on correctly identified interactions
   → Classes: 0=Minor, 1=Moderate, 2=Major
   → Measures quality of risk assessment
""")
    
    print("\n" + "="*70)
    print(" CLINICAL SIGNIFICANCE")
    print("="*70)
    
    print("""
🏥 For Patient Safety:
   • HIGH RECALL is most critical - we cannot afford to miss dangerous 
     interactions (false negatives can harm patients)
   
   • HIGH PRECISION reduces alert fatigue - too many false alarms cause
     healthcare providers to ignore warnings
   
   • HIGH SEVERITY ACCURACY ensures critical interactions are flagged
     appropriately for urgent action

📊 Trade-offs:
   • Can increase recall by lowering threshold → more false alarms
   • Can increase precision by raising threshold → miss more interactions
   • F1 score helps find optimal balance
""")
    
    print("\n" + "="*70)
    print(" TO SEE ACTUAL VALUES")
    print("="*70)
    print("""
The model has been trained and metrics were printed during training.
To get current metrics, you can:

1. Check training logs (if saved during original training)
2. Run full evaluation (computationally intensive):
   python get_metrics.py

3. The model automatically evaluates on test set after training
   and prints results in format:

   FINAL TEST RESULTS
   ==========================================
   AUC: 0.XXXX
   ACCURACY: 0.XXXX
   PRECISION: 0.XXXX
   RECALL: 0.XXXX
   F1: 0.XXXX
   SEVERITY_ACC: 0.XXXX
""")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    show_metrics_info()
