"""Compile .proto files to Python gRPC stubs.

Usage:
    python -m server.grpc.compile_protos
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def compile_protos() -> None:
    """Run protoc via grpc_tools to generate Python stubs."""
    grpc_dir = Path(__file__).parent
    protos_dir = grpc_dir / "protos"
    output_dir = grpc_dir / "generated"
    output_dir.mkdir(exist_ok=True)

    proto_files = list(protos_dir.glob("*.proto"))
    if not proto_files:
        print("No .proto files found in", protos_dir)
        return

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={protos_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",
    ] + [str(p.name) for p in proto_files]

    print(f"Compiling {len(proto_files)} proto file(s)...")
    print(f"  Proto dir: {protos_dir}")
    print(f"  Output dir: {output_dir}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("protoc error:", result.stderr)
        sys.exit(1)

    # Fix relative imports in generated files (grpc_tools generates
    # `import common_pb2` but we need `from . import common_pb2`)
    for py_file in output_dir.glob("*_pb2_grpc.py"):
        text = py_file.read_text(encoding="utf-8")
        fixed = text.replace(
            "import common_pb2",
            "from . import common_pb2",
        ).replace(
            "import render_pb2",
            "from . import render_pb2",
        )
        if fixed != text:
            py_file.write_text(fixed, encoding="utf-8")
            print(f"  Fixed imports: {py_file.name}")

    for py_file in output_dir.glob("*_pb2.py"):
        text = py_file.read_text(encoding="utf-8")
        fixed = text.replace(
            "import common_pb2",
            "from . import common_pb2",
        )
        if fixed != text:
            py_file.write_text(fixed, encoding="utf-8")
            print(f"  Fixed imports: {py_file.name}")

    print("Done. Generated stubs in:", output_dir)


if __name__ == "__main__":
    compile_protos()
