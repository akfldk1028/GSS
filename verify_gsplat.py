# Verify gsplat imports and CUDA functionality
import sys

print("Python:", sys.version)
print("-" * 50)

# Test gsplat import
try:
    import gsplat
    print("PASS: gsplat imported successfully")
except Exception as e:
    print("FAIL: gsplat import -", str(e))
    sys.exit(1)

# Test gsplat.rasterization import (requires CUDA)
try:
    from gsplat import rasterization
    print("PASS: gsplat.rasterization imported successfully")
    print("OK")
except Exception as e:
    print("FAIL: gsplat.rasterization import -", str(e))
    sys.exit(1)
