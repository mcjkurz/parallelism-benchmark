#!/usr/bin/env python3
"""
CUDA diagnostic script to test PyTorch GPU setup
"""
import sys

print("=" * 60)
print("CUDA DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: PyTorch import
print("\n[1] Importing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"✗ Failed to import PyTorch: {e}")
    sys.exit(1)

# Test 2: CUDA built version
print("\n[2] Checking CUDA build info...")
print(f"   PyTorch built with CUDA: {torch.version.cuda}")
print(f"   cuDNN version: {torch.backends.cudnn.version()}")

# Test 3: CUDA availability
print("\n[3] Checking CUDA availability...")
try:
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    if not cuda_available:
        print("   ✗ CUDA is not available!")
        print("   This is the main issue - PyTorch cannot access CUDA")
except Exception as e:
    print(f"   ✗ Error checking CUDA: {e}")
    sys.exit(1)

# Test 4: Device count
print("\n[4] Checking GPU devices...")
try:
    device_count = torch.cuda.device_count()
    print(f"   Device count: {device_count}")
    if device_count == 0:
        print("   ✗ No CUDA devices found!")
except Exception as e:
    print(f"   ✗ Error getting device count: {e}")

# Test 5: Device properties
print("\n[5] GPU device details...")
try:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"   Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
except Exception as e:
    print(f"   ✗ Error getting device properties: {e}")

# Test 6: Try to create a tensor on GPU
print("\n[6] Testing GPU tensor creation...")
try:
    if torch.cuda.is_available():
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = x * 2
        print(f"   ✓ Successfully created and computed on GPU: {y.cpu().tolist()}")
    else:
        print("   ⊗ Skipped (CUDA not available)")
except Exception as e:
    print(f"   ✗ Error creating GPU tensor: {e}")

# Test 7: Check environment
print("\n[7] Environment check...")
import os
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

