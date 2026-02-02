import os
import argparse
import subprocess
from pathlib import Path
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate images with different PKLs and steps")
    
    # 核心参数
    parser.add_argument("--networks", nargs="+", help="List of .pkl network files", required=True)
    parser.add_argument("--gen_steps", nargs="+", type=int, default=[18], help="List of generation steps to test (e.g. 18 50 100)")
    
    # 生成控制参数
    parser.add_argument("--seeds", type=str, default="0-63", help="Random seeds (e.g. 0-63)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    
    # 输出设置
    parser.add_argument("--out_root", type=str, default="batch_results", help="Root directory for outputs")
    
    # 其他 generate.py 的参数 (根据之前对话中的参数设置默认值)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--on_latents", type=int, default=1, help="1 for True, 0 for False")
    parser.add_argument("--S_churn", type=float, default=40)
    parser.add_argument("--S_min", type=float, default=0.05)
    parser.add_argument("--S_max", type=float, default=50)
    parser.add_argument("--S_noise", type=float, default=1.003)
    
    return parser.parse_args()

def extract_train_step(filename):
    """
    尝试从文件名中提取训练步数。
    例如: 'network-snapshot-005000.pkl' -> '005000'
    如果找不到数字，返回原始文件名。
    """
    # 匹配文件名末尾的数字序列
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    return filename

def main():
    args = parse_args()
    
    # 确保输出根目录存在
    os.makedirs(args.out_root, exist_ok=True)
    
    # 构造基础命令
    # 注意: generate.py 需要在当前工作目录下
    base_cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={args.gpu_count}", "generate.py",
        "--batch", str(args.batch),
        "--seeds", args.seeds,
        "--resolution", str(args.resolution),
        f"--S_churn={args.S_churn}",
        f"--S_min={args.S_min}",
        f"--S_max={args.S_max}",
        f"--S_noise={args.S_noise}",
        "--subdirs" # 保持子目录结构
    ]
    
    if args.on_latents:
        base_cmd.append("--on_latents=1")
    
    print(f"=== Starting Batch Generation ===")
    print(f"Networks: {args.networks}")
    print(f"Steps: {args.gen_steps}")
    print(f"GPUs: {args.gpu_count}")
    print("===============================\n")

    for pkl_path in args.networks:
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            print(f"[Warning] Network file not found: {pkl_path}, skipping...")
            continue
            
        pkl_name = pkl_path.stem
        # 提取训练步数用于命名 (如果文件名比较长，可以只提取步数部分简化目录名，或者直接用完整文件名)
        # 这里为了清晰，使用 "文件名_生成步数" 的格式
        
        for step in args.gen_steps:
            # 构造输出目录: {out_root}/{pkl_name}_step{step}
            # 例如: batch_results/network-snapshot-005000_genstep50
            folder_name = f"{pkl_name}_genstep{step}"
            out_dir = os.path.join(args.out_root, folder_name)
            
            print(f"--> Processing: {pkl_name}")
            print(f"    Gen Steps: {step}")
            print(f"    Output: {out_dir}")
            
            # 组合完整命令
            cmd = base_cmd + [
                f"--network={pkl_path}",
                f"--steps={step}",
                f"--outdir={out_dir}"
            ]
            
            # 打印命令
            print(f"    Executing: {' '.join(cmd)}")
            
            try:
                subprocess.run(cmd, check=True)
                print(f"    [Success] Finished {folder_name}\n")
            except subprocess.CalledProcessError as e:
                print(f"    [Error] Failed processing {folder_name}. Error: {e}\n")
            except Exception as e:
                print(f"    [Error] Unexpected error: {e}\n")

if __name__ == "__main__":
    main()
