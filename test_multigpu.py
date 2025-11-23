#!/usr/bin/env python3
"""
Quick test to verify multi-GPU detection and basic setup.
Run this on your Lambda instance before starting full training.
"""

import torch

def test_gpu_detection():
    print("=" * 60)
    print("GPU Detection Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Running on CPU.")
        return False
    
    print("✓ CUDA is available")
    
    # Count GPUs
    n_gpus = torch.cuda.device_count()
    print(f"✓ Found {n_gpus} GPU(s)")
    
    # List all GPUs
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"  GPU {i}: {name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")
    
    print(f"✓ CUDA version: {torch.version.cuda}")
    
    # Test basic tensor operations on each GPU
    print("\nTesting tensor operations on each GPU...")
    for i in range(n_gpus):
        device = f"cuda:{i}"
        try:
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = x @ y  # Matrix multiply
            result = z.mean().item()
            print(f"  GPU {i}: ✓ (test result: {result:.4f})")
        except Exception as e:
            print(f"  GPU {i}: ❌ Error: {e}")
            return False
    
    # Test multi-GPU transfer
    if n_gpus > 1:
        print("\nTesting inter-GPU transfers...")
        try:
            x = torch.randn(1000, 1000, device="cuda:0")
            x_copy = x.to("cuda:1")
            print(f"  ✓ Successfully transferred tensor from GPU 0 → GPU 1")
            
            # Verify data integrity
            x_back = x_copy.to("cuda:0")
            if torch.allclose(x, x_back):
                print(f"  ✓ Data integrity verified")
            else:
                print(f"  ❌ Data mismatch after transfer!")
                return False
        except Exception as e:
            print(f"  ❌ Transfer error: {e}")
            return False
    
    # Estimate memory available for training
    print("\nMemory availability:")
    for i in range(n_gpus):
        device = f"cuda:{i}"
        torch.cuda.set_device(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        free = total - reserved
        print(f"  GPU {i}: {free:.1f} GB free of {total:.1f} GB total")
    
    print("\n" + "=" * 60)
    if n_gpus >= 8:
        print("✅ All 8 GPUs detected and working!")
        print("You're ready for fast multi-GPU training.")
    elif n_gpus > 1:
        print(f"✅ {n_gpus} GPUs detected and working!")
        print("Multi-GPU training will be enabled.")
    else:
        print("✅ 1 GPU detected and working!")
        print("Training will use single GPU mode.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_gpu_detection()
    exit(0 if success else 1)

