# Test all acceptance criteria imports
import sys
print("Python:", sys.version)
print("-" * 50)

# Test 1: torch
try:
    import torch
    print("PASS: torch", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("FAIL: torch -", str(e)[:100])

# Test 2: gsplat
try:
    import gsplat
    print("PASS: gsplat")
except Exception as e:
    print("FAIL: gsplat -", str(e)[:100])

# Test 3: matplotlib
try:
    import matplotlib
    print("PASS: matplotlib")
except Exception as e:
    print("FAIL: matplotlib -", str(e)[:100])

# Test 4: gss
try:
    import gss
    print("PASS: gss")
except Exception as e:
    print("FAIL: gss -", str(e)[:100])
