"""
METIS Inference Server — OpenAI-compatible API with Cognitive Metadata.

Provides a minimal HTTP server for METIS inference with:
  - POST /v1/chat/completions  — OpenAI-compatible endpoint
  - GET  /health               — Health check

Non-streaming mode uses generate_cognitive() for DPO-trained models.
Streaming mode uses generate() with on_token callback for real-time SSE.

Usage:
    python -m metis.serve --model experiment_output_dpo_balanced/metis_dpo_cognitive
    # Then:
    curl -X POST http://localhost:8741/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "What is 5+7?"}]}'

Requires: pip install starlette uvicorn
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict

logger = logging.getLogger("metis.serve")


def create_app(
    model_path: str,
    device: str = "auto",
) -> Any:
    """Create the ASGI application with METIS inference engine."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    # Lazy-loaded engine state
    _state: Dict[str, Any] = {}

    def _ensure_engine() -> None:
        if "engine" in _state:
            return

        import torch
        from metis.metis import Metis
        from metis.inference import MetisInference

        logger.info(f"Loading model: {model_path}")
        metis = Metis.from_pretrained(model_path)
        engine = MetisInference(metis)
        _state["engine"] = engine
        _state["metis"] = metis
        _state["model_path"] = model_path
        logger.info("Model loaded, server ready.")

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "model": _state.get("model_path", "not_loaded"),
        })

    async def chat_completions(request: Request) -> Any:
        _ensure_engine()
        engine: "MetisInference" = _state["engine"]

        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 1024)

        # Extract last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        if not prompt:
            return JSONResponse(
                {"error": "No user message found in messages"},
                status_code=400,
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = _state.get("model_path", "metis")

        if stream:
            return StreamingResponse(
                _stream_generate(engine, prompt, max_tokens, request_id, model_name),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: use generate_cognitive
        result = engine.generate_cognitive(
            prompt, max_new_tokens=max_tokens,
        )

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result.tokens_generated,
                "total_tokens": result.tokens_generated,
            },
            "metis": {
                "cognitive_route": result.cognitive_route,
                "final_decision": result.final_decision.value
                if hasattr(result.final_decision, "value")
                else str(result.final_decision),
                "thinking_text": result.thinking_text,
                "thinking_repaired": result.thinking_repaired,
                "latency_ms": round(result.latency_ms, 1),
            },
        }
        return JSONResponse(response)

    async def _stream_generate(
        engine: Any,
        prompt: str,
        max_tokens: int,
        request_id: str,
        model_name: str,
    ) -> AsyncIterator[str]:
        """SSE streaming via generate() with on_token callback."""
        import asyncio
        import queue

        token_queue: queue.Queue[Optional[str]] = queue.Queue()

        def on_token(token_text: str, signal: Any) -> None:
            token_queue.put(token_text)

        # Run generation in a thread to avoid blocking the event loop
        import threading

        def _run() -> None:
            try:
                engine.generate(
                    prompt,
                    max_tokens=max_tokens,
                    on_token=on_token,
                )
            except Exception as e:
                logger.error(f"Stream generation error: {e}")
            finally:
                token_queue.put(None)  # Sentinel

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            try:
                token = token_queue.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if token is None:
                # Final chunk
                done_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ]

    return Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="METIS Inference Server — OpenAI-compatible API"
    )
    parser.add_argument(
        "--model",
        default="experiment_output_dpo_balanced/metis_dpo_cognitive",
        help="Model path",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8741, help="Bind port")
    parser.add_argument("--device", default="auto", help="Device")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = create_app(model_path=args.model, device=args.device)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
