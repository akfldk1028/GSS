"""Diagnostic script for PyTorch DLL loading issues on Windows."""
import sys
import os
import ctypes.util

print("=" * 60)
print("PyTorch DLL Diagnostics")
print("=" * 60)

# Check Python version
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")

# Check if fbgemm.dll exists
torch_lib = r"C:\Users\User\anaconda3\envs\gss\Lib\site-packages\torch\lib"
fbgemm_path = os.path.join(torch_lib, "fbgemm.dll")
c10_path = os.path.join(torch_lib, "c10.dll")

print(f"\nDLL Checks:")
print(f"  fbgemm.dll exists: {os.path.exists(fbgemm_path)}")
print(f"  c10.dll exists: {os.path.exists(c10_path)}")

if os.path.exists(fbgemm_path):
    print(f"  fbgemm.dll size: {os.path.getsize(fbgemm_path)} bytes")

# Check system PATH
print(f"\nCUDA paths in PATH:")
path_dirs = os.environ.get('PATH', '').split(';')
cuda_paths = [p for p in path_dirs if 'cuda' in p.lower()]
for p in cuda_paths:
    print(f"  {p}")

# Check for VC++ redistributables
print(f"\nVisual C++ Runtime Check:")
vc_dlls = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll']
for dll in vc_dlls:
    found = ctypes.util.find_library(dll.replace('.dll', ''))
    print(f"  {dll}: {'✓ FOUND' if found else '✗ NOT FOUND'}")

print("\nRecommendation:")
if not any(ctypes.util.find_library(dll.replace('.dll', '')) for dll in vc_dlls):
    print("  Install Visual C++ 2015-2022 Redistributable (x64)")
    print("  Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
else:
    print("  VC++ runtime found - issue may be CUDA-related")
    print("  Try CPU-only PyTorch or different CUDA version")

# Try to get more info from Windows
try:
    import subprocess
    result = subprocess.run(['where', 'cudart64_*.dll'], capture_output=True, text=True, shell=True)
    if result.stdout:
        print(f"\nCUDA Runtime DLLs found:")
        print(result.stdout)
except:
    pass
