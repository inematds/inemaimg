"""Runtime shims for NVIDIA Blackwell / GB10 (sm_120).

NGC pytorch:25.03-py3 ships a PyTorch build whose JIT-reduction path hands
nvrtc an arch value that the bundled nvrtc rejects with:

    nvrtc: error: invalid value for --gpu-architecture (-arch)

This bites us specifically on `Tensor.prod(-1)` over an int64 CUDA tensor,
triggered deep inside transformers' Qwen2.5-VL image encoder:

    # transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1226
    split_sizes = (image_grid_thw.prod(-1) // ...).tolist()

`image_grid_thw` is a tiny metadata tensor (3 ints). Routing int CUDA tensors
through CPU for `prod`/`cumprod` avoids the JIT path entirely and costs
microseconds. Float tensors are left untouched because those ops are
precompiled and work correctly.

Remove this file once the container base image is bumped past this issue
(NGC 25.04+ or a newer torch wheel built for sm_120 end-to-end).
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("inemaimg.shims")

_INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def _patch_int_reductions() -> None:
    """Route int-CUDA prod/cumprod through CPU to dodge the nvrtc JIT path."""
    _orig_prod = torch.Tensor.prod
    _orig_cumprod = torch.Tensor.cumprod

    def _safe_prod(self, *args, **kwargs):
        if self.is_cuda and self.dtype in _INT_DTYPES:
            return _orig_prod(self.cpu(), *args, **kwargs).to(self.device)
        return _orig_prod(self, *args, **kwargs)

    def _safe_cumprod(self, *args, **kwargs):
        if self.is_cuda and self.dtype in _INT_DTYPES:
            return _orig_cumprod(self.cpu(), *args, **kwargs).to(self.device)
        return _orig_cumprod(self, *args, **kwargs)

    torch.Tensor.prod = _safe_prod  # type: ignore[method-assign]
    torch.Tensor.cumprod = _safe_cumprod  # type: ignore[method-assign]
    log.info("applied Blackwell nvrtc shim: int-CUDA prod/cumprod → CPU")


_patch_int_reductions()
