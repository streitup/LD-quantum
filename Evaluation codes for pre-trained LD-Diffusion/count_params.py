import torch
import pickle
import sys
import os

# Function to count params in a pickle snapshot
def count_params(pkl_path):
    print(f"Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # EDM checkpoints usually have 'ema' or 'model'
        # data is usually a dict with keys like 'G_ema', 'G', 'D', etc.
        # We focus on the Generator (G_ema or G)
        
        model = data.get('ema', data.get('G_ema', data.get('model', None)))
        if model is None:
            print(f"Could not find model key in {list(data.keys())}")
            return
            
        total_params = 0
        trainable_params = 0
        
        # Recursively count params
        for name, param in model.named_parameters():
            num = param.numel()
            total_params += num
            if param.requires_grad:
                trainable_params += num
                
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
        
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_params.py <pkl_path>")
        sys.exit(1)
        
    pkl_path = sys.argv[1]
    count_params(pkl_path)
