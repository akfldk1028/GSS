"""Omniverse Nucleus integration — USD Live Sync.

Provides connection management and file push to Nucleus server,
enabling real-time digital twin updates in Isaac Sim / Omniverse.

Requires: omni-client package (from NVIDIA Omniverse).
Falls back gracefully when not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _has_omni_client() -> bool:
    try:
        import omni.client  # noqa: F401
        return True
    except ImportError:
        return False


class NucleusConnection:
    """Manage connection to an Omniverse Nucleus server."""

    def __init__(self, server_url: str):
        """
        Args:
            server_url: Nucleus server URL, e.g. "omniverse://localhost"
        """
        self.server_url = server_url.rstrip("/")
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to Nucleus server."""
        if not _has_omni_client():
            logger.warning(
                "omni.client not installed — Nucleus sync disabled. "
                "Install Omniverse Connect SDK for live sync."
            )
            return False

        try:
            import omni.client

            result = omni.client.stat(self.server_url)
            if result[0] == omni.client.Result.OK:
                self._connected = True
                logger.info("Connected to Nucleus: %s", self.server_url)
                return True
            else:
                logger.error("Nucleus connection failed: %s", result[0])
                return False
        except Exception as e:
            logger.error("Nucleus connection error: %s", e)
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def push_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload a local file to Nucleus.

        Args:
            local_path: Local file path (e.g. processed/model.usdc)
            remote_path: Nucleus path (e.g. /Projects/GSS/job123/model.usdc)

        Returns:
            True if successful.
        """
        if not self._connected:
            logger.warning("Not connected to Nucleus, skipping push")
            return False

        full_remote = f"{self.server_url}{remote_path}"

        try:
            import omni.client

            result = omni.client.copy(str(local_path), full_remote)
            if result == omni.client.Result.OK:
                logger.info("Pushed %s → %s", local_path.name, full_remote)
                return True
            else:
                logger.error("Nucleus push failed: %s", result)
                return False
        except Exception as e:
            logger.error("Nucleus push error: %s", e)
            return False

    def create_live_session(self, stage_path: str) -> str | None:
        """Create or join a Live Session for collaborative editing.

        Args:
            stage_path: Nucleus path to the USD stage.

        Returns:
            Session name if successful, None otherwise.
        """
        if not self._connected:
            return None

        try:
            import omni.client

            full_path = f"{self.server_url}{stage_path}"

            # Create the live layer path
            live_path = full_path.replace(".usdc", ".live").replace(".usd", ".live")

            # Check if live layer exists
            result = omni.client.stat(live_path)
            if result[0] != omni.client.Result.OK:
                # Create empty live layer
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".live", delete=False) as tmp:
                    tmp.write(b"#usda 1.0\n")
                    tmp_path = tmp.name
                omni.client.copy(tmp_path, live_path)
                Path(tmp_path).unlink(missing_ok=True)

            session_name = Path(stage_path).stem
            logger.info("Live session ready: %s", live_path)
            return session_name

        except Exception as e:
            logger.error("Live session error: %s", e)
            return None

    def push_usd_delta(
        self,
        stage_path: str,
        prim_path: str,
        attributes: dict[str, Any],
    ) -> bool:
        """Push incremental USD attribute updates via the live layer.

        This enables real-time scene updates without replacing the full file.

        Args:
            stage_path: Nucleus path to the USD stage.
            prim_path: USD prim path (e.g. "/World/Walls/Wall_001")
            attributes: Dict of attribute name → value to update.
        """
        if not self._connected:
            return False

        try:
            from pxr import Sdf, Usd

            full_path = f"{self.server_url}{stage_path}"
            live_path = full_path.replace(".usdc", ".live").replace(".usd", ".live")

            # Open or create the live layer
            layer = Sdf.Layer.FindOrOpen(live_path)
            if not layer:
                layer = Sdf.Layer.CreateNew(live_path)

            # Apply delta
            with Sdf.ChangeBlock():
                prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
                for attr_name, value in attributes.items():
                    attr_spec = Sdf.AttributeSpec(
                        prim_spec, attr_name, Sdf.ValueTypeNames.Find(type(value).__name__)
                    )
                    attr_spec.default = value

            layer.Save()
            logger.info("USD delta pushed: %s [%s]", prim_path, ", ".join(attributes.keys()))
            return True

        except Exception as e:
            logger.error("USD delta error: %s", e)
            return False


class NucleusManager:
    """Application-level Nucleus manager — singleton per server instance."""

    def __init__(self) -> None:
        self._connection: NucleusConnection | None = None

    def configure(self, server_url: str) -> bool:
        """Initialize and connect to Nucleus."""
        self._connection = NucleusConnection(server_url)
        return self._connection.connect()

    @property
    def connection(self) -> NucleusConnection | None:
        return self._connection

    @property
    def is_available(self) -> bool:
        return self._connection is not None and self._connection.is_connected

    def push_job_outputs(self, job_id: str, job_dir: Path, project_path: str = "/Projects/GSS") -> dict[str, bool]:
        """Push all USD outputs from a job to Nucleus.

        Args:
            job_id: Job identifier.
            job_dir: Local job directory.
            project_path: Nucleus project root path.

        Returns:
            Dict of filename → success status.
        """
        if not self.is_available:
            return {}

        results = {}
        processed = job_dir / "processed"
        if not processed.exists():
            return results

        for f in processed.rglob("*"):
            if f.suffix.lower() in (".usdc", ".usd", ".usdz"):
                remote = f"{project_path}/{job_id}/{f.relative_to(processed).as_posix()}"
                results[f.name] = self._connection.push_file(f, remote)

        return results
