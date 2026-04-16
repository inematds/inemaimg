"""Loader for black-forest-labs/FLUX.2-klein-9B.

Step-distilled rectified-flow transformer (9B params, text encoder Qwen3 8B).
Generates in 4 steps, supports both text-to-image and edit with up to 4
reference images. We wire in FLUX.2-small-decoder (Apache 2.0) for ~1.4x
faster VAE decode at zero quality cost.

Per BFL's official docs:
- guidance_scale is IGNORED by the distilled checkpoint — always set to 1.0.
- negative_prompt is NOT supported by the FLUX.2 pipeline architecture.
- Prompt style: prose ("describe like a novelist"), not keyword lists.
  Quality boosters like "masterpiece", "8k" have no effect.
- caption_upsample_temperature=0.15 improves short prompts automatically.
- Resolutions: 128–2048px in 16px increments, baseline 1024².

See docs/flux2-klein.md for the full prompting and parameter guide.

WARNING: FLUX Non-Commercial license. Disable in REGISTRY for commercial
deployments without a BFL agreement.
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
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MAX_IMAGES = 4

    @classmethod
    def load(cls):
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
        if len(images) > cls.MAX_IMAGES:
            raise ValueError(
                f"flux2-klein accepts at most {cls.MAX_IMAGES} reference images."
            )

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

        width = int(req.width) if req.width is not None else cls.DEFAULT_WIDTH
        height = int(req.height) if req.height is not None else cls.DEFAULT_HEIGHT
        width = max(128, min(2048, (width // 16) * 16))
        height = max(128, min(2048, (height // 16) * 16))

        kwargs: dict = dict(
            prompt=req.prompt,
            width=width,
            height=height,
            num_inference_steps=req.steps if req.steps is not None else cls.DEFAULT_STEPS,
            guidance_scale=1.0,
        )
        # caption_upsample_temperature improves short prompts but is only
        # available in recent diffusers builds. Try it, skip if unsupported.
        try:
            import inspect
            sig = inspect.signature(pipe.__class__.__call__)
            if "caption_upsample_temperature" in sig.parameters:
                kwargs["caption_upsample_temperature"] = 0.15
        except Exception:
            pass
        if images:
            kwargs["image"] = images if len(images) > 1 else images[0]
        if generator is not None:
            kwargs["generator"] = generator
        # negative_prompt intentionally NOT passed — FLUX.2 pipeline
        # does not support it and would raise.

        result = pipe(**kwargs)
        return result.images[0]
