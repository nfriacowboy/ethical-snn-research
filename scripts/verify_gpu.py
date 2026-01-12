#!/usr/bin/env python3
"""Verify GPU availability for PyTorch."""

import sys


def main():
    print("=" * 60)
    print("GPU Verification for Ethical SNN Research")
    print("=" * 60)
    print()

    # Check PyTorch
    try:
        import torch

        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")
        sys.exit(1)

    # Check CUDA/ROCm availability
    print(f"\nCUDA/ROCm available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")

            # Memory info
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  Total memory: {mem_total:.2f} GB")

        # Test tensor creation
        print("\nTesting tensor operations...")
        try:
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            print("✓ GPU tensor operations successful")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")

    else:
        print("\n⚠ No GPU detected - simulations will run on CPU")
        print("  This will be significantly slower")
        print("  Consider installing ROCm (AMD) or CUDA (NVIDIA)")

    # Check snntorch
    try:
        import snntorch

        print(f"\n✓ snnTorch version: {snntorch.__version__}")
    except ImportError:
        print("\n✗ snnTorch not installed")
        print("  Install with: pip install snntorch")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
