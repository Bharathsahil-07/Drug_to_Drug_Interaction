
import torch
import os
from torch_geometric.data import Data

def inspect():
    path = 'data/trained_model_v2.pt'
    print(f"Inspecting {path}...")
    
    if not os.path.exists(path):
        print("File does not exist")
        return

    try:
        ckpt = torch.load(path, weights_only=False)
        print(f"Type: {type(ckpt)}")
        
        if isinstance(ckpt, dict):
            print(f"Keys: {list(ckpt.keys())}")
            if 'model_state_dict' in ckpt:
                print(f"State dict type: {type(ckpt['model_state_dict'])}")
        else:
            print(f"Content: {ckpt}")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    inspect()
