"""Server configuration via environment variables (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    """All tunables for the GSS API server.

    Values are read from environment variables (prefixed GSS_)
    or from a .env file in the project root.
    """

    model_config = {"env_prefix": "GSS_", "env_file": ".env", "extra": "ignore"}

    # --- paths ---
    data_root: Path = Path("./data")
    jobs_dir: Path = Path("./data/jobs")
    upload_max_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB

    # --- pipeline ---
    default_pipeline_config: Path = Path("./configs/pipeline_import.yaml")

    # --- worker ---
    worker_count: int = 1  # GPU workers (typically 1 per GPU)

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Phase 3: Nucleus ---
    nucleus_server: str = ""  # e.g. "omniverse://192.168.1.100"
    nucleus_project_path: str = "/Projects/GSS"

    # --- Phase 4: MQTT ---
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_topics: str = "gss/sensors/#"  # comma-separated

    # --- gRPC Render Server ---
    grpc_enabled: bool = True          # Start gRPC render server in lifespan
    grpc_port: int = 46435             # NuRec convention port
    grpc_max_models: int = 4           # Max concurrent models in GPU memory

    # --- gRPC TLS (disabled by default — development mode) ---
    grpc_tls_enabled: bool = False
    grpc_tls_cert: str = "server/grpc/certs/server.crt"
    grpc_tls_key: str = "server/grpc/certs/server.key"
    grpc_tls_ca: str = ""              # CA cert for mutual TLS (empty = no mTLS)

    # --- gRPC Security: checkpoint path whitelist ---
    grpc_allowed_checkpoint_dirs: str = "./data,./clone/3dgrut"  # comma-separated


settings = ServerSettings()
