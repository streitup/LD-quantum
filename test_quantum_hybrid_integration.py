
import torch
import sys
import os

# Add parent directory to path to find training module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Training Codes of LD-Diffusion')))

from training.networks import SongUNet

def test_integration():
    print("Testing Quantum Hybrid Integration...")
    
    # Configuration
    img_resolution = 32
    in_channels = 4 # Latent space
    out_channels = 4
    model_channels = 32
    channel_mult = [1, 2, 2] # 32, 64, 64
    num_blocks = 2
    attn_resolutions = [16, 8] # Apply attention at 16x16 and 8x8
    
    # Quantum Adapter Config
    quantum_adapter = "training.quantum_transformer:QuantumAdapterHybrid"
    quantum_adapter_kwargs = {"device_name": "cpu"} # Use CPU for test
    
    # Instantiate Model
    print("Instantiating SongUNet...")
    # Note: SongUNet internally sets channel_mult, so we check what resolutions get attention.
    # 32x32 -> 32ch (mult 1) -> 32ch. num_heads=32//64=0. No attention.
    # 16x16 -> 64ch (mult 2) -> 64ch. num_heads=64//64=1. Attention!
    # 8x8 -> 64ch (mult 2) -> 64ch. num_heads=64//64=1. Attention!
    
    model = SongUNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        use_quantum_transformer=True,
        quantum_adapter=quantum_adapter,
        quantum_adapter_kwargs=quantum_adapter_kwargs
    )
    
    print("Model instantiated successfully.")
    
    # Check if adapters are present in UNetBlocks
    print("\nChecking UNetBlocks for Quantum Adapters:")
    found_adapters = 0
    checked_blocks = 0
    
    for name, module in model.named_modules():
        # UNetBlocks are usually named like 'enc.16x16.block0' or similar structure in SongUNet
        # SongUNet has .enc, .dec, .middle
        if hasattr(module, 'quantum_adapter'): # It's a UNetBlock
            checked_blocks += 1
            if module.quantum_adapter is not None:
                print(f"  - {name}: Found adapter {type(module.quantum_adapter).__name__}")
                print(f"    - Block Channels: {module.out_channels}")
                print(f"    - Block Num Heads: {module.num_heads}")
                
                # Check adapter properties
                adapter = module.quantum_adapter
                print(f"    - Adapter Num Heads: {adapter.num_heads}")
                print(f"    - Adapter Input Dim: {adapter.in_channels}")
                
                # Verify consistency
                if adapter.num_heads != module.num_heads:
                     print(f"    WARNING: Adapter num_heads ({adapter.num_heads}) != Block num_heads ({module.num_heads})")
                
                if hasattr(adapter, 'attn'):
                    print(f"    - Inner Attn Input Dim: {adapter.attn.input_dim_val}")
                    # qk_dim verification
                    if hasattr(adapter.attn, 'qk_dim'):
                        print(f"    - Inner Attn qk_dim: {adapter.attn.qk_dim}")
                        expected_qk = adapter.in_channels // adapter.num_heads
                        if adapter.attn.qk_dim != expected_qk:
                             print(f"    ERROR: qk_dim mismatch! Expected {expected_qk}, got {adapter.attn.qk_dim}")
                        else:
                             print(f"    - qk_dim correct ({expected_qk})")
                    
                found_adapters += 1
            else:
                # If no adapter, check if it was expected
                # If num_heads > 0 and attention=True, it should have one.
                # Inspect block properties
                if module.num_heads > 0 and getattr(module, 'use_quantum_transformer', False):
                     # Wait, we set use_quantum_transformer=False if instantiation failed.
                     # But if it wasn't even attempted (e.g. num_heads=0), use_quantum_transformer might still be True?
                     # Let's check.
                     pass
                if module.num_heads == 0:
                     # print(f"  - {name}: No adapter (num_heads=0)")
                     pass
                else:
                     print(f"  - {name}: No adapter (num_heads={module.num_heads})")

    
    if found_adapters > 0:
        print(f"\nSuccess: Found {found_adapters} quantum adapters initialized correctly.")
    else:
        print("\nFailure: No quantum adapters found.")
        
    # Forward Pass Test
    print("\nRunning Forward Pass...")
    x = torch.randn(2, in_channels, img_resolution, img_resolution) # Batch size 2
    t = torch.randn(2)
    class_labels = None
    
    try:
        out = model(x, t, class_labels)
        print(f"Forward pass successful. Output shape: {out.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
