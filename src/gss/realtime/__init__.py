"""gss.realtime — Real-time 3DGS capture, training, and export.

Architecture:
    FrameBuffer (capture.py)     — Threaded camera → frame accumulation
    IncrementalTrainer (trainer.py) — Fast gsplat training from frames
    USDZExporter (exporter.py)   — PLY → USDZ via 3DGRUT
    RealtimeOrchestrator (orchestrator.py) — Main loop tying all together
"""

from .config import RealtimeConfig

__all__ = ["RealtimeConfig"]
