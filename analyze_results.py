import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze():
    files = glob.glob('ablation_results_*.json')
    if not files:
        print("No result files found yet.")
        return

    results = {}
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            results.update(data)

    print(f"\nLoaded results for {len(results)} models.")
    
    # Sort keys to match the order
    keys = sorted(results.keys())
    
    print("\n" + "="*100)
    print(f"{'Model Name':<30} | {'PSNR (dB)':<15} | {'SSIM':<15} | {'LPIPS':<15} | {'Loss':<10}")
    print("-" * 100)
    
    baseline_psnr = 0
    if "1. Baseline: QSANN" in results:
        baseline_psnr = results["1. Baseline: QSANN"]["final_psnr"]
        
    for k in keys:
        res = results[k]
        psnr = res['final_psnr']
        ssim = res['final_ssim']
        lpips = res['final_lpips']
        loss = res['final_loss']
        
        # Check for std deviation
        std_psnr = res.get('std_psnr', 0)
        std_ssim = res.get('std_ssim', 0)
        std_lpips = res.get('std_lpips', 0)
        
        gain = ""
        if "Baseline" not in k and baseline_psnr > 0:
            diff = psnr - baseline_psnr
            gain = f"({diff:+.2f})"
            
        psnr_str = f"{psnr:.2f}±{std_psnr:.2f}" if std_psnr > 0 else f"{psnr:.2f}"
        ssim_str = f"{ssim:.4f}±{std_ssim:.4f}" if std_ssim > 0 else f"{ssim:.4f}"
        lpips_str = f"{lpips:.2f}±{std_lpips:.2f}" if std_lpips > 0 else f"{lpips:.4f}"
        
        # Adjust PSNR string with gain
        if gain:
            psnr_str += f" {gain}"
            
        print(f"{k:<30} | {psnr_str:<15} | {ssim_str:<15} | {lpips_str:<15} | {loss:.4f}")
    print("="*100 + "\n")

    # Plotting convergence curves if available
    # Assuming the json contains history (metrics list)
    # The current script saves the whole 'results' dict which includes logs?
    # Let's check the structure in benchmark_sota_ablation.py
    # Yes, it saves "metrics": log_metrics
    
    plt.figure(figsize=(15, 10))
    
    # PSNR Plot
    plt.subplot(2, 2, 1)
    for k in keys:
        if 'metrics' in results[k]:
            plt.plot(results[k]['metrics']['psnr'], label=k)
    plt.title('PSNR Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(2, 2, 2)
    for k in keys:
        if 'metrics' in results[k]:
            plt.plot(results[k]['metrics']['loss'], label=k)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # LPIPS Plot
    plt.subplot(2, 2, 3)
    for k in keys:
        if 'metrics' in results[k]:
            plt.plot(results[k]['metrics']['lpips'], label=k)
    plt.title('LPIPS Score (Lower is better)')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ablation_analysis.png')
    print("Analysis plot saved to ablation_analysis.png")

if __name__ == "__main__":
    analyze()
