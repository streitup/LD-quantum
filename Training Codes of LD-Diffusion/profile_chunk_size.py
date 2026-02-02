import torch
import torch.nn as nn
import time
import os
import sys
import gc

# Ensure current path is in sys.path
sys.path.append(os.getcwd())

from training.networks import DhariwalUNet

def profile_chunk_size():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on: {device}")
    
    # Common settings (simulate training load)
    B, C, H, W = 4, 128, 32, 32 
    noise_labels = torch.randn(B).to(device)
    raw_labels = torch.randint(0, 10, (B,)).to(device)
    class_labels = torch.zeros(B, 10).to(device).scatter_(1, raw_labels.unsqueeze(1), 1.0)
    x = torch.randn(B, C, H, W).to(device)
    
    # Chunk sizes to test
    chunk_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    results = []
    
    print("\n" + "="*60)
    print(f"{'Chunk Size':<15} | {'Throughput (it/s)':<20} | {'Time/Iter (ms)':<15} | {'Max Mem (MB)':<15}")
    print("="*60)
    
    for cs in chunk_sizes:
        try:
            # Force GC
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Instantiate model with specific chunk size
            model = DhariwalUNet(
                img_resolution=H,
                in_channels=C,
                out_channels=C,
                label_dim=10,
                model_channels=128,
                channel_mult=[1, 1],
                num_blocks=1,
                use_qcnn_frontend=True,
                qcnn_chunk_size=cs, 
                qcnn_use_strided=False,
                qcnn_reupload=True
            ).to(device)
            
            # Compile optimization?
            # We can try to compile the frontend manually here to test if it helps
            # if hasattr(model.enc['32x32_block0'].quantum_frontend, '_apply_fusion_circuit'):
            #     model.enc['32x32_block0'].quantum_frontend._apply_fusion_circuit = torch.compile(
            #         model.enc['32x32_block0'].quantum_frontend._apply_fusion_circuit, mode="max-autotune"
            #     )
            # But compilation takes time (warmup).
            
            # Warmup
            for _ in range(3):
                _ = model(x, noise_labels, class_labels)
            
            # Measure
            torch.cuda.synchronize()
            start = time.time()
            steps = 10
            for _ in range(steps):
                _ = model(x, noise_labels, class_labels)
            torch.cuda.synchronize()
            
            avg_time = (time.time() - start) / steps
            throughput = 1.0 / avg_time
            max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            print(f"{cs:<15} | {throughput:<20.2f} | {avg_time*1000:<15.2f} | {max_mem:<15.2f}")
            results.append((cs, throughput))
            
            del model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{cs:<15} | {'OOM':<20} | {'N/A':<15} | {'N/A':<15}")
            else:
                print(f"{cs:<15} | {f'Error: {e}':<20}")
            break # Stop if OOM or error
            
    best_cs = max(results, key=lambda x: x[1])[0] if results else 4096
    print("="*60)
    print(f"Best Chunk Size found: {best_cs}")

if __name__ == "__main__":
    profile_chunk_size()
