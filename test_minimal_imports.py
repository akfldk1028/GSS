# Test 1: Can we import torch?
try:
    import torch
    print(f"✓ torch {torch.__version__} imported successfully")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ torch import failed: {e}")

# Test 2: Can we import gsplat?
try:
    import gsplat
    print(f"✓ gsplat imported successfully")
except Exception as e:
    print(f"✗ gsplat import failed: {e}")

# Test 3: Can we import matplotlib?
try:
    import matplotlib
    print(f"✓ matplotlib imported successfully")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")

# Test 4: Can we import gss?
try:
    import gss
    print(f"✓ gss imported successfully")
except Exception as e:
    print(f"✗ gss import failed: {e}")
