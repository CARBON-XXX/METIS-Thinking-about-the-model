"""
METIS Signal Bridge — WebSocket real-time broadcast.

Runs a lightweight WebSocket server on a background thread.
METIS.step() pushes signals via the registered callback;
the bridge broadcasts JSON to all connected dashboard clients.

Usage:
    from metis.bridge import SignalBridge

    bridge = SignalBridge(port=8765)
    bridge.start()

    metis.add_listener(bridge.on_signal)
    # ... run training ...
    bridge.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from queue import Queue, Empty
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .metis import Metis
    from .core.types import CognitiveSignal

logger = logging.getLogger(__name__)

# Optional: try websockets, fall back to None
try:
    import websockets
    import websockets.server
    _HAS_WS = True
except ImportError:
    _HAS_WS = False


class SignalBridge:
    """Real-time WebSocket bridge for METIS cognitive signals."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        if not _HAS_WS:
            raise ImportError(
                "websockets package required: pip install websockets"
            )
        self._host = host
        self._port = port
        self._queue: Queue[str] = Queue(maxsize=2048)
        self._clients: set[Any] = set()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

        # Training progress metadata (set externally)
        self.prompt_index: int = 0
        self.sample_index: int = 0
        self.total_prompts: int = 300
        self.current_prompt: str = ""
        self.phase: str = "generate"

    # ─── Public API ───

    def start(self) -> None:
        """Start WebSocket server on background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="metis-bridge")
        self._thread.start()
        logger.info(f"[Bridge] WebSocket server started on ws://{self._host}:{self._port}")

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("[Bridge] WebSocket server stopped")

    def on_signal(self, signal: "CognitiveSignal", metis: "Metis") -> None:
        """
        Callback for Metis.add_listener(). Serializes signal + stats → queue.
        """
        try:
            ctrl_stats = metis.stats.get("controller", {})
            msg = {
                "type": "signal",
                "signal": {
                    "semantic_entropy": round(signal.semantic_entropy, 4),
                    "token_entropy": round(signal.token_entropy, 4),
                    "semantic_diversity": round(signal.semantic_diversity, 4),
                    "confidence": round(signal.confidence, 4),
                    "z_score": round(signal.z_score, 4),
                    "decision": signal.decision.value if hasattr(signal.decision, 'value') else str(signal.decision),
                    "entropy_trend": signal.entropy_trend,
                    "cognitive_phase": signal.cognitive_phase,
                    "entropy_momentum": round(signal.entropy_momentum, 4),
                    "token_surprise": round(signal.token_surprise, 4),
                    "boundary_action": signal.boundary_action.value if hasattr(signal.boundary_action, 'value') else str(signal.boundary_action),
                    "cusum_alarm": signal.cusum_alarm,
                },
                "controller": {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in ctrl_stats.items()
                },
                "meta": {
                    "prompt_index": self.prompt_index,
                    "sample_index": self.sample_index,
                    "total_prompts": self.total_prompts,
                    "current_prompt": self.current_prompt[:80],
                    "phase": self.phase,
                },
            }
            # Non-blocking put — drop if queue full (dashboard can miss frames)
            try:
                self._queue.put_nowait(json.dumps(msg))
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[Bridge] Signal serialization error: {e}")

    # ─── Internal ───

    def _run(self) -> None:
        """Background thread: run asyncio event loop with WebSocket server."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        """Main async: start server + broadcast loop."""
        async with websockets.server.serve(
            self._handler, self._host, self._port,
            ping_interval=20, ping_timeout=10,
        ):
            logger.info(f"[Bridge] Listening on ws://{self._host}:{self._port}")
            while not self._stop_event.is_set():
                await self._broadcast_queued()
                await asyncio.sleep(0.02)  # ~50 Hz max broadcast rate

    async def _handler(self, websocket: Any) -> None:
        """Handle a WebSocket client connection."""
        self._clients.add(websocket)
        addr = websocket.remote_address
        logger.info(f"[Bridge] Client connected: {addr}")
        try:
            async for _ in websocket:
                pass  # We don't expect messages from dashboard
        finally:
            self._clients.discard(websocket)
            logger.info(f"[Bridge] Client disconnected: {addr}")

    async def _broadcast_queued(self) -> None:
        """Drain queue and broadcast to all clients."""
        messages = []
        while len(messages) < 20:  # Max 20 messages per broadcast cycle
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                break

        if not messages or not self._clients:
            return

        dead = set()
        for msg in messages:
            for client in self._clients:
                try:
                    await client.send(msg)
                except Exception:
                    dead.add(client)
        self._clients -= dead
