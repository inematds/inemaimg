"""Loader for Qwen/Qwen-Image-Edit-2511.

This is the commercial motor of the project (Apache 2.0). It's an edit model:
takes 1..3 reference images plus a prompt. We do NOT call
enable_model_cpu_offload() — on GB10 / DGX Spark the GPU already reads from
the unified LPDDR5X, so offload only adds redundant copies. If you re-deploy
this loader to a discrete-VRAM host (e.g. RTX 4090, 24 GB), flip the
USE_CPU_OFFLOAD flag below.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

if TYPE_CHECKING:
    from server import GenerateRequest


USE_CPU_OFFLOAD = False  # True only on discrete-VRAM hosts with <40GB


class QwenEditLoader:
    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

    # Defaults, per the Qwen model card.
    DEFAULT_STEPS = 40
    DEFAULT_TRUE_CFG = 4.0
    DEFAULT_GUIDANCE = 1.0
    DEFAULT_NEGATIVE = " "  # single space, per model card

    @classmethod
    def load(cls):
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            cls.MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        if USE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
        return pipe

    @classmethod
    def generate(
        cls,
        pipe,
        req: "GenerateRequest",
        images: list[Image.Image],
    ) -> Image.Image:
        if not images:
            raise ValueError(
                "qwen-edit-2511 is an edit model and requires at least one "
                "reference image in the 'images' field."
            )
        if len(images) > 3:
            raise ValueError(
                "qwen-edit-2511 accepts at most 3 reference images."
            )

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

        kwargs: dict = dict(
            image=images,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or cls.DEFAULT_NEGATIVE,
            true_cfg_scale=req.true_cfg_scale if req.true_cfg_scale is not None else cls.DEFAULT_TRUE_CFG,
            guidance_scale=req.guidance_scale if req.guidance_scale is not None else cls.DEFAULT_GUIDANCE,
            num_inference_steps=req.steps if req.steps is not None else cls.DEFAULT_STEPS,
        )
        if req.width is not None:
            kwargs["width"] = int(req.width)
        if req.height is not None:
            kwargs["height"] = int(req.height)
        if generator is not None:
            kwargs["generator"] = generator

        result = pipe(**kwargs)
        return result.images[0]
