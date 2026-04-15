"""Loader for diffusers/FLUX.2-dev-bnb-4bit.

32B-param FLUX.2-dev, bitsandbytes 4-bit quantized so it fits in 24 GB of
discrete VRAM. On GB10 with 119 GB unified we could run it unquantized, but
the bnb-4bit build loads in seconds and preserves enough quality for hero
renders. We also swap in FLUX.2-small-decoder (Apache 2.0) for the VAE.

WARNING: FLUX.2-dev is FLUX Non-Commercial. Same caveat as klein.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKLFlux2, Flux2Pipeline
from PIL import Image

if TYPE_CHECKING:
    from server import GenerateRequest


class Flux2DevLoader:
    MODEL_ID = "diffusers/FLUX.2-dev-bnb-4bit"
    DECODER_ID = "black-forest-labs/FLUX.2-small-decoder"

    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE = 4.0
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    @classmethod
    def load(cls):
        vae = AutoencoderKLFlux2.from_pretrained(
            cls.DECODER_ID,
            torch_dtype=torch.bfloat16,
        )
        # text_encoder=None: the bnb-4bit build is designed to be paired with
        # an external text encoder. The pipeline still works for simple prompts
        # via its internal tokenizer path. If you need higher-fidelity text
        # conditioning, load T5 separately and pass it in.
        pipe = Flux2Pipeline.from_pretrained(
            cls.MODEL_ID,
            vae=vae,
            text_encoder=None,
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
        if len(images) > 3:
            raise ValueError("flux2-dev accepts at most 3 reference images.")

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

        kwargs: dict = dict(
            prompt=req.prompt,
            width=int(req.width) if req.width is not None else cls.DEFAULT_WIDTH,
            height=int(req.height) if req.height is not None else cls.DEFAULT_HEIGHT,
            num_inference_steps=req.steps if req.steps is not None else cls.DEFAULT_STEPS,
            guidance_scale=req.guidance_scale if req.guidance_scale is not None else cls.DEFAULT_GUIDANCE,
        )
        if images:
            kwargs["image"] = images if len(images) > 1 else images[0]
        if req.negative_prompt is not None:
            kwargs["negative_prompt"] = req.negative_prompt
        if generator is not None:
            kwargs["generator"] = generator

        result = pipe(**kwargs)
        return result.images[0]
