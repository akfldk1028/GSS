"""gRPC security utilities — TLS credentials and checkpoint path validation.

Two concerns:
1. TLS: Encrypt gRPC traffic in production (optional, disabled by default)
2. Checkpoint validation: Prevent arbitrary file loading via torch.load()
   which can execute arbitrary Python code (pickle deserialization attack)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Checkpoint path whitelist ─────────────────────────────────────

# Directories from which checkpoint loading is allowed.
# Resolved to absolute paths at validation time to prevent symlink traversal.
_DEFAULT_ALLOWED_DIRS: list[str] = [
    "./data",           # Pipeline outputs (jobs, interim)
    "./clone/3dgrut",   # 3DGRUT training outputs
]


def validate_checkpoint_path(
    checkpoint_path: str,
    allowed_dirs: list[str] | None = None,
) -> Path:
    """Validate that a checkpoint path is within allowed directories.

    Prevents path traversal attacks (e.g. "../../etc/passwd") and
    restricts torch.load() to trusted directories only.

    Args:
        checkpoint_path: Path to the checkpoint file.
        allowed_dirs: List of allowed parent directories.
                      Defaults to _DEFAULT_ALLOWED_DIRS.

    Returns:
        Resolved absolute Path if valid.

    Raises:
        PermissionError: If path is outside allowed directories.
        FileNotFoundError: If checkpoint doesn't exist.
    """
    if allowed_dirs is None:
        allowed_dirs = _DEFAULT_ALLOWED_DIRS

    resolved = Path(checkpoint_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    if not resolved.is_file():
        raise PermissionError(f"Checkpoint path is not a file: {resolved}")

    # Check suffix
    allowed_suffixes = {".pt", ".pth", ".ckpt"}
    if resolved.suffix.lower() not in allowed_suffixes:
        raise PermissionError(
            f"Invalid checkpoint extension '{resolved.suffix}'. "
            f"Allowed: {allowed_suffixes}"
        )

    # Check against allowed directories
    allowed_resolved = [Path(d).resolve() for d in allowed_dirs]
    for allowed in allowed_resolved:
        try:
            resolved.relative_to(allowed)
            logger.debug("Checkpoint '%s' validated under '%s'", resolved, allowed)
            return resolved
        except ValueError:
            continue

    raise PermissionError(
        f"Checkpoint path '{resolved}' is outside allowed directories: "
        f"{[str(d) for d in allowed_resolved]}. "
        f"Set GSS_GRPC_ALLOWED_CHECKPOINT_DIRS to add more."
    )


# ── TLS credentials ──────────────────────────────────────────────


def load_tls_server_credentials(
    cert_path: str | Path,
    key_path: str | Path,
    ca_path: str | Path | None = None,
) -> "grpc.ServerCredentials":
    """Load TLS credentials for the gRPC server.

    Args:
        cert_path: Path to server certificate (PEM).
        key_path: Path to server private key (PEM).
        ca_path: Path to CA certificate for mutual TLS (optional).

    Returns:
        grpc.ServerCredentials for secure port binding.
    """
    import grpc

    cert_path = Path(cert_path)
    key_path = Path(key_path)

    if not cert_path.exists():
        raise FileNotFoundError(f"TLS cert not found: {cert_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"TLS key not found: {key_path}")

    server_cert = cert_path.read_bytes()
    server_key = key_path.read_bytes()

    root_ca = None
    require_client_auth = False
    if ca_path:
        ca_path = Path(ca_path)
        if ca_path.exists():
            root_ca = ca_path.read_bytes()
            require_client_auth = True
            logger.info("Mutual TLS enabled (client cert required)")

    credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=root_ca,
        require_client_auth=require_client_auth,
    )
    logger.info("TLS server credentials loaded from %s", cert_path)
    return credentials


def load_tls_channel_credentials(
    ca_path: str | Path,
    client_cert_path: str | Path | None = None,
    client_key_path: str | Path | None = None,
) -> "grpc.ChannelCredentials":
    """Load TLS credentials for the gRPC client.

    Args:
        ca_path: Path to CA certificate for server verification (PEM).
        client_cert_path: Path to client certificate for mutual TLS (optional).
        client_key_path: Path to client private key for mutual TLS (optional).

    Returns:
        grpc.ChannelCredentials for secure channel creation.
    """
    import grpc

    ca_path = Path(ca_path)
    if not ca_path.exists():
        raise FileNotFoundError(f"CA cert not found: {ca_path}")

    root_cert = ca_path.read_bytes()

    client_cert = None
    client_key = None
    if client_cert_path and client_key_path:
        client_cert = Path(client_cert_path).read_bytes()
        client_key = Path(client_key_path).read_bytes()

    credentials = grpc.ssl_channel_credentials(
        root_certificates=root_cert,
        private_key=client_key,
        certificate_chain=client_cert,
    )
    return credentials
