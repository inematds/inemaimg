"""inemaimg — multi-model image server for timesmkt3.

FastAPI gateway with a hot-swap model registry. Starts with a single model
(qwen-edit-2511) and is designed to grow into ERNIE + FLUX.2 once the MVP is
validated on the GB10 / DGX Spark host.

Concurrency model: one asyncio lock around the GPU. Only one request touches
the pipeline at a time — guarantees we never OOM the GPU by running two
generations in parallel, and serializes model swaps.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from loaders.qwen_edit import QwenEditLoader

log = logging.getLogger("inemaimg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

# Internal id -> loader class. Add new models here as they come online.
REGISTRY: dict[str, type] = {
    "qwen-edit-2511": QwenEditLoader,
}


class _State:
    loaded_id: str | None = None
    pipe: Any | None = None
    lock: asyncio.Lock | None = None
    # Prewarm status, used by /health so orchestrators can tell "booting" from "stuck".
    prewarm_status: str = "idle"  # idle | running | ready | failed
    prewarm_error: str | None = None


state = _State()


def _load(model_id: str) -> None:
    """Synchronous load/swap. Caller must hold state.lock."""
    if state.loaded_id == model_id and state.pipe is not None:
        return
    _unload()
    loader_cls = REGISTRY[model_id]
    log.info("loading model %s (%s)", model_id, loader_cls.__name__)
    t0 = time.perf_counter()
    state.pipe = loader_cls.load()
    state.loaded_id = model_id
    log.info("loaded %s in %.1fs", model_id, time.perf_counter() - t0)


def _unload() -> None:
    if state.pipe is None:
        return
    log.info("unloading model %s", state.loaded_id)
    del state.pipe
    state.pipe = None
    state.loaded_id = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def _run_prewarm(model_id: str) -> None:
    """Load a model off the event loop so /health stays responsive during boot."""
    loop = asyncio.get_running_loop()
    assert state.lock is not None
    async with state.lock:
        state.prewarm_status = "running"
        state.prewarm_error = None
        try:
            await loop.run_in_executor(None, _load, model_id)
            state.prewarm_status = "ready"
        except Exception as e:  # pragma: no cover — diagnostic path
            log.exception("prewarm of %s failed: %s", model_id, e)
            state.prewarm_status = "failed"
            state.prewarm_error = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.lock = asyncio.Lock()
    prewarm = os.environ.get("INEMAIMG_PREWARM", "").strip()
    prewarm_task: asyncio.Task | None = None
    if prewarm:
        if prewarm not in REGISTRY:
            log.warning("INEMAIMG_PREWARM=%s not in registry, skipping prewarm", prewarm)
        else:
            # Fire-and-forget: lifespan yields immediately so /health is live
            # while the download + GPU copy happen in the background. Callers
            # can poll /health for prewarm_status until it's "ready".
            prewarm_task = asyncio.create_task(_run_prewarm(prewarm))
    yield
    if prewarm_task is not None and not prewarm_task.done():
        prewarm_task.cancel()
    _unload()


app = FastAPI(title="inemaimg", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class GenerateRequest(BaseModel):
    model: str = Field(..., description="Internal model id from REGISTRY")
    prompt: str
    images: list[str] | None = Field(
        default=None,
        description="Reference images as base64-encoded PNG/JPEG (edit models only).",
    )
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    true_cfg_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    # Optional LoRA name — recognized by loaders that support them
    # (e.g. qwen-edit-2511 supports 'multiple-angles' and 'face-swap')
    lora: str | None = None
    lora_weight: float | None = None


class LoadRequest(BaseModel):
    model: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _decode_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated() / (1024 ** 3), 2)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_model": state.loaded_id,
        "prewarm_status": state.prewarm_status,
        "prewarm_error": state.prewarm_error,
        "gpu_memory_allocated_gb": _gpu_mem_gb(),
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/models")
def models():
    return {
        "available": list(REGISTRY.keys()),
        "loaded": state.loaded_id,
    }


@app.post("/models/load")
async def load_model(req: LoadRequest):
    if req.model not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown model: {req.model}")
    assert state.lock is not None
    async with state.lock:
        _load(req.model)
    return {"loaded": state.loaded_id, "gpu_memory_allocated_gb": _gpu_mem_gb()}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if req.model not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown model: {req.model}")
    # Decode images outside the lock — pure CPU work, no need to serialize.
    pil_images = [_decode_image(b) for b in (req.images or [])]

    assert state.lock is not None
    async with state.lock:
        _load(req.model)
        loader_cls = REGISTRY[req.model]
        t0 = time.perf_counter()
        try:
            image = loader_cls.generate(state.pipe, req, pil_images)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("generation failed")
            raise HTTPException(status_code=500, detail=f"generation failed: {e}")
        dt = time.perf_counter() - t0

    return {
        "image": _encode_image(image),
        "model_used": req.model,
        "generation_time_s": round(dt, 2),
        "gpu_memory_allocated_gb": _gpu_mem_gb(),
    }
