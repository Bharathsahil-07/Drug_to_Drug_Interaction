
import torch
import os
import sys
from torch_geometric.data import Data

def package_model():
    print("Packaging best_model.pt into trained_model_v2.pt...")
    
    try:
        # Load raw state dict
        if not os.path.exists('data/best_model.pt'):
            print("Error: data/best_model.pt not found")
            return

        state_dict = torch.load('data/best_model.pt', weights_only=False)
        
        # Load graph for metadata
        if not os.path.exists('data/drug_graph_v2.pt'):
             print("Error: data/drug_graph_v2.pt not found")
             return

        graph_dict = torch.load('data/drug_graph_v2.pt', weights_only=False)
        graph_data = graph_dict['graph_data']
        drug_to_idx = graph_dict['drug_to_idx']
        idx_to_drug = graph_dict['idx_to_drug']
        
        input_dim = graph_data.x.shape[1]
        
        # Config from train_v2.py
        config = {
            'hidden_dim': 128,
            'embedding_dim': 64,
            'heads': 4
        }
        
        checkpoint = {
            'model_state_dict': state_dict,
            'drug_to_idx': drug_to_idx,
            'idx_to_drug': idx_to_drug,
            'input_dim': input_dim,
            'config': config
        }
        
        torch.save(checkpoint, 'data/trained_model_v2.pt')
        print(f"✅ Package complete. Saved to data/trained_model_v2.pt")
        
    except Exception as e:
        print(f"❌ Error packaging model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    package_model()
