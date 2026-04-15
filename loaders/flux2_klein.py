"""Loader for black-forest-labs/FLUX.2-klein-9B.

Step-distilled rectified-flow transformer (9B params). Generates in 4 steps,
supports both text-to-image and edit (single or multi-reference). We also
wire in FLUX.2-small-decoder (Apache 2.0) because it's strictly better than
the default decoder — ~1.4x faster at zero quality cost.

WARNING: FLUX.2 models are FLUX Non-Commercial license. Keep this model
disabled in REGISTRY if the deployment is part of a commercial product
without a BFL agreement.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from PIL import Image

if TYPE_CHECKING:
    from server import GenerateRequest


class Flux2KleinLoader:
    MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
    DECODER_ID = "black-forest-labs/FLUX.2-small-decoder"

    DEFAULT_STEPS = 4
    DEFAULT_GUIDANCE = 1.0  # klein is step-distilled, guidance is fixed
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    @classmethod
    def load(cls):
        # Swap in the small VAE decoder — strictly faster, same quality.
        vae = AutoencoderKLFlux2.from_pretrained(
            cls.DECODER_ID,
            torch_dtype=torch.bfloat16,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            cls.MODEL_ID,
            vae=vae,
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
            raise ValueError("flux2-klein accepts at most 3 reference images.")

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

        kwargs: dict = dict(
            prompt=req.prompt,
            width=int(req.width) if req.width is not None else cls.DEFAULT_WIDTH,
            height=int(req.height) if req.height is not None else cls.DEFAULT_HEIGHT,
            num_inference_steps=req.steps if req.steps is not None else cls.DEFAULT_STEPS,
            guidance_scale=cls.DEFAULT_GUIDANCE,  # klein ignores caller override
        )
        if images:
            # FLUX.2 takes a single image or a list — the diffusers wrapper
            # accepts both shapes.
            kwargs["image"] = images if len(images) > 1 else images[0]
        if req.negative_prompt is not None:
            kwargs["negative_prompt"] = req.negative_prompt
        if generator is not None:
            kwargs["generator"] = generator

        result = pipe(**kwargs)
        return result.images[0]
