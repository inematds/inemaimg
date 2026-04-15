"""Loader for baidu/ERNIE-Image.

Pure text-to-image DiT (~8B params). No image-to-image, no reference input,
no ControlNet. We expose it as the commercial T2I fallback alongside
Qwen-Edit — both have open licenses (Apache 2.0 / ERNIE open).

GB10 has unified memory, so we load straight onto CUDA without CPU offload.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers import ErnieImagePipeline
from PIL import Image

if TYPE_CHECKING:
    from server import GenerateRequest


class ErnieLoader:
    MODEL_ID = "baidu/ERNIE-Image"

    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 4.0
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    @classmethod
    def load(cls):
        pipe = ErnieImagePipeline.from_pretrained(
            cls.MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        return pipe

    @classmethod
    def generate(
        cls,
        pipe,
        req: "GenerateRequest",
        images: list[Image.Image],
    ) -> Image.Image:
        if images:
            raise ValueError(
                "ernie is a pure text-to-image model and does not accept "
                "reference images — remove the 'images' field."
            )

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

        kwargs: dict = dict(
            prompt=req.prompt,
            width=int(req.width) if req.width is not None else cls.DEFAULT_WIDTH,
            height=int(req.height) if req.height is not None else cls.DEFAULT_HEIGHT,
            num_inference_steps=req.steps if req.steps is not None else cls.DEFAULT_STEPS,
            guidance_scale=req.guidance_scale if req.guidance_scale is not None else cls.DEFAULT_GUIDANCE,
            use_pe=True,
        )
        if req.negative_prompt is not None:
            kwargs["negative_prompt"] = req.negative_prompt
        if generator is not None:
            kwargs["generator"] = generator

        result = pipe(**kwargs)
        return result.images[0]
