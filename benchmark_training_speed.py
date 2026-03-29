
import torch
import time
import os
import sys
import numpy as np

# Add path to find training modules
sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))

from training.networks import SongUNet, DhariwalUNet

def benchmark_speed(device='cuda'):
    print(f"=== Training Speed Benchmark: Quantum vs Classical ===")
    print(f"Device: {device}")
    
    # Configuration mimicking /date/zzn_data/quantum-panda-32/00004-panda_32-uncond-ncsnpp-edm-gpus1-batch32-fp32
    # Based on path:
    # - ncsnpp (SongUNet)
    # - edm (EDMPrecond -> SongUNet)
    # - batch32
    # - fp32
    # - resolution 32
    
    img_resolution = 32
    in_channels = 3
    out_channels = 3
    label_dim = 0
    model_channels = 128 # Default for SongUNet
    channel_mult = [1, 2, 2, 2] # Default
    attn_resolutions = [16]
    batch_size = 1 # Drastically reduced to 1 to fit in memory
    
    # Dummy Input
    x = torch.randn(batch_size, in_channels, img_resolution, img_resolution).to(device)
    noise_labels = torch.randn(batch_size).to(device) # Random noise levels
    class_labels = None
    
    configs = [
        ("Classical (Baseline)", False),
        ("Quantum SOTA (Enabled)", True)
    ]
    
    results = []
    
    for name, use_quantum in configs:
        print(f"\n--- Benchmarking {name} ---")
        
        try:
            model = SongUNet(
                img_resolution=img_resolution,
                in_channels=in_channels,
                out_channels=out_channels,
                label_dim=label_dim,
                model_channels=model_channels,
                channel_mult=channel_mult,
                attn_resolutions=attn_resolutions,
                use_qcnn_frontend=use_quantum,
                qcnn_resolutions=None, # All resolutions if enabled
                qcnn_chunk_size=128 # Limit chunk size to save memory
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Warmup
            print("  Warmup (3 iters)...")
            for _ in range(3):
                out = model(x, noise_labels, class_labels)
                loss = out.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            n_iter = 20 # Enough to get stable avg
            print(f"  Running {n_iter} iterations...")
            
            start_event.record()
            for _ in range(n_iter):
                out = model(x, noise_labels, class_labels)
                loss = out.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_ms = start_event.elapsed_time(end_event)
            avg_time = elapsed_ms / n_iter
            
            # Memory Check
            mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            print(f"  Avg Time: {avg_time:.2f} ms/iter")
            print(f"  Peak Mem: {mem_alloc:.2f} MB")
            
            results.append({
                "name": name,
                "time": avg_time,
                "mem": mem_alloc
            })
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Final Results ===")
    print(f"{'Model':<25} | {'Time (ms/iter)':<15} | {'Speedup':<10} | {'Memory (MB)':<12}")
    print("-" * 70)
    
    base_time = results[0]['time']
    for r in results:
        speedup = base_time / r['time']
        print(f"{r['name']:<25} | {r['time']:<15.2f} | {speedup:<10.2f}x | {r['mem']:<12.2f}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_speed()
    else:
        print("CUDA not available, skipping speed benchmark.")
