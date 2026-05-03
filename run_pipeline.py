"""
Drug Interaction Project - Main Runner
Run the complete pipeline: Parse â†’ Build Graph â†’ Train MT-GAT
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required libraries are installed"""
    required = ['torch', 'pandas', 'numpy', 'sklearn', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"âš ï¸  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def run_pipeline(xml_file, max_drugs=None, skip_parse=False, skip_graph=False):
    """
    Run the complete pipeline
    
    Args:
        xml_file: Path to DrugBank XML file
        max_drugs: Limit number of drugs (for testing)
        skip_parse: Skip parsing if data already exists
        skip_graph: Skip graph building if already exists
    """
    
    print("="*70)
    print("DRUG INTERACTION PREDICTION PIPELINE")
    print("Using Multi-Task Graph Attention Network (MT-GAT)")
    print("="*70)
    
    # Check if XML file exists
    if not os.path.exists(xml_file):
        print(f"\nâŒ Error: XML file not found: {xml_file}")
        return
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Step 1: Parse XML
    if not skip_parse or not os.path.exists('data/drugs.csv'):
        print("\n" + "="*70)
        print("STEP 1: PARSING DRUGBANK XML")
        print("="*70)
        
        from data_parser import DrugBankParser
        
        parser = DrugBankParser(xml_file)
        drugs_df, interactions_df = parser.parse_all(max_drugs=max_drugs)
        parser.save_to_csv(drugs_df, interactions_df)
        
        print("\nâœ… Parsing complete!")
    else:
        print("\nâœ… Using existing parsed data (data/drugs.csv)")
    
    # Step 2: Build Graph
    if not skip_graph or not os.path.exists('data/drug_graph.pt'):
        print("\n" + "="*70)
        print("STEP 2: BUILDING DRUG INTERACTION GRAPH")
        print("="*70)
        
        from graph_builder import DrugGraphBuilder
        import pandas as pd
        
        drugs_df = pd.read_csv('data/drugs.csv')
        interactions_df = pd.read_csv('data/interactions.csv')
        
        builder = DrugGraphBuilder(drugs_df, interactions_df)
        graph_data = builder.build_graph(use_text_features=True, max_text_features=50)
        builder.save_graph('data/drug_graph.pt')
        
        # Show statistics
        stats = builder.get_statistics()
        print("\nGraph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\nâœ… Graph building complete!")
    else:
        print("\nâœ… Using existing graph (data/drug_graph.pt)")
    
    # Step 3: Train MT-GAT
    print("\n" + "="*70)
    print("STEP 3: TRAINING MT-GAT MODEL")
    print("="*70)
    
    from mt_gat_model import DrugInteractionMTGAT, MTGATTrainer
    from graph_builder import DrugGraphBuilder
    import torch
    
    # Load graph
    graph_data, drug_to_idx, idx_to_drug = DrugGraphBuilder.load_graph('data/drug_graph.pt')
    
    # Check if CUDA available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionMTGAT(
        input_dim=input_dim,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    trainer = MTGATTrainer(model, graph_data, device=device)
    trainer.split_edges(train_ratio=0.8, val_ratio=0.1)
    
    print("\nStarting training... (this may take a while)")
    test_metrics = trainer.train(epochs=200, lr=0.001)
    
    # Plot training curves
    trainer.plot_training()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'drug_to_idx': drug_to_idx,
        'idx_to_drug': idx_to_drug,
        'input_dim': input_dim,
        'test_metrics': test_metrics
    }, 'data/trained_model.pt')
    
    print("\n" + "="*70)
    print("âœ… PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  ðŸ“„ data/drugs.csv - Parsed drug data")
    print("  ðŸ“„ data/interactions.csv - Parsed interactions")
    print("  ðŸ“Š data/drug_graph.pt - Drug interaction graph")
    print("  ðŸ¤– data/trained_model.pt - Trained MT-GAT model")
    print("  ðŸ“ˆ data/training_curves.png - Training visualization")
    print("\nTest Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    print("\nYou can now use the model for predictions!")


def quick_test():
    """Quick test with trained model"""
    print("\n" + "="*70)
    print("QUICK TEST: PREDICTING INTERACTIONS")
    print("="*70)
    
    import torch
    import pandas as pd
    from mt_gat_model import DrugInteractionMTGAT
    from graph_builder import DrugGraphBuilder
    
    # Load model
    checkpoint = torch.load('data/trained_model.pt')
    
    input_dim = checkpoint['input_dim']
    model = DrugInteractionMTGAT(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    drug_to_idx = checkpoint['drug_to_idx']
    idx_to_drug = checkpoint['idx_to_drug']
    
    # Load graph
    graph_data, _, _ = DrugGraphBuilder.load_graph('data/drug_graph.pt')
    
    # Load drug names
    drugs_df = pd.read_csv('data/drugs.csv')
    
    # Test some drug pairs
    print("\nTesting drug pairs:")
    
    test_pairs = [
        ('DB00945', 'DB00564'),  # Common interaction
        ('DB00001', 'DB00002'),  # First two drugs
    ]
    
    for drug1_id, drug2_id in test_pairs:
        if drug1_id in drug_to_idx and drug2_id in drug_to_idx:
            drug1_name = drugs_df[drugs_df['drug_id'] == drug1_id]['name'].values[0]
            drug2_name = drugs_df[drugs_df['drug_id'] == drug2_id]['name'].values[0]
            
            prob = model.predict_interaction(
                graph_data.x,
                graph_data.edge_index,
                drug_to_idx[drug1_id],
                drug_to_idx[drug2_id]
            )
            
            print(f"\n  {drug1_name} â†” {drug2_name}")
            print(f"  Interaction probability: {prob:.2%}")
            print(f"  Risk: {'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.5 else 'LOW'}")


def main():
    """Main entry point"""
    
    # Check requirements
    if not check_requirements():
        print("\nâŒ Please install required packages first")
        return
    
    # Configuration
    xml_file = r"c:\Users\navas\Downloads\New folder (10)\full database.xml"
    
    print("\nðŸ”¬ Drug Interaction Prediction Pipeline")
    print("\nOptions:")
    print("  1. Quick test (100 drugs)")
    print("  2. Full dataset (all drugs)")
    print("  3. Custom number of drugs")
    print("  4. Test existing model")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        print("\nðŸš€ Running quick test with 100 drugs...")
        run_pipeline(xml_file, max_drugs=100)
    
    elif choice == '2':
        confirm = input("\nâš ï¸  This will take 30-60 minutes. Continue? (y/n): ")
        if confirm.lower() == 'y':
            print("\nðŸš€ Running full pipeline...")
            run_pipeline(xml_file)
        else:
            print("Cancelled.")
    
    elif choice == '3':
        try:
            num_drugs = int(input("Number of drugs: "))
            print(f"\nðŸš€ Running pipeline with {num_drugs} drugs...")
            run_pipeline(xml_file, max_drugs=num_drugs)
        except ValueError:
            print("Invalid number")
    
    elif choice == '4':
        if os.path.exists('data/trained_model.pt'):
            quick_test()
        else:
            print("\nâŒ No trained model found. Please train first.")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nâš ï¸  Interrupted by user")
    except Exception as e:
        print(f"\nâŒ Error: {e}")
        import traceback
        traceback.print_exc()

