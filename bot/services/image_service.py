"""
Athena — Image Generation Service

Wraps Stable Diffusion (via diffusers).
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler
from PIL import Image

from bot.config import settings

logger = logging.getLogger("athena.image")


@dataclass
class ImageResult:
    path: Path
    seed: int
    width: int
    height: int
    elapsed_seconds: float


class ImageService:
    """Manages Stable Diffusion model and inference."""

    def __init__(self) -> None:
        self._pipeline: Optional[AutoPipelineForText2Image] = None
        self._loaded = False
        self._device = f"cuda:{settings.sd_gpu_device}" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load the model into VRAM."""
        if self._loaded:
            return

        logger.info("Loading Stable Diffusion model from %s ...", settings.sd_model_id)
        try:
            # Load pipeline
            # Use fp16 for GPU to save VRAM, float32 for CPU
            dtype = torch.float16 if "cuda" in self._device else torch.float32
            
            # Helper to load from local cache if possible, else download
            model_id_or_path = settings.sd_model_id
            
            # Specific optimization for Flux models (4-bit quantization)
            if "flux" in model_id_or_path.lower() and "cuda" in self._device:
                try:
                    from transformers import BitsAndBytesConfig, T5EncoderModel
                    from diffusers import FluxTransformer2DModel
                    
                    logger.info("Detected Flux model: Enabling 4-bit quantization for memory optimization...")
                    
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                    )
                    
                    # Load Transformer in 4-bit
                    transformer = FluxTransformer2DModel.from_pretrained(
                        model_id_or_path,
                        subfolder="transformer",
                        quantization_config=quant_config,
                        torch_dtype=dtype,
                        cache_dir=settings.sd_model_local_path,
                    )
                    
                    # Load T5 Encoder in 4-bit (saves additional ~6GB)
                    text_encoder_2 = T5EncoderModel.from_pretrained(
                        model_id_or_path,
                        subfolder="text_encoder_2",
                        quantization_config=quant_config,
                        torch_dtype=dtype,
                        cache_dir=settings.sd_model_local_path,
                    )
                    
                    self._pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_id_or_path,
                        transformer=transformer,
                        text_encoder_2=text_encoder_2,
                        torch_dtype=dtype,
                        cache_dir=settings.sd_model_local_path,
                    )
                    logger.info("✓ Quantized Flux model loaded successfully")
                    
                except ImportError:
                     logger.warning("bitsandbytes not found, falling back to standard loading")
                     self._pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_id_or_path,
                        torch_dtype=dtype,
                        cache_dir=settings.sd_model_local_path,
                    )
            else:
                self._pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id_or_path,
                    torch_dtype=dtype,
                    cache_dir=settings.sd_model_local_path,
                )

            # Optimizations
            # Optimizations
            self._pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self._pipeline.scheduler.config
            )
            
            # CPU Offload strategy
            # If 4-bit quantization is enabled (Flux), the model fits in VRAM (approx 10GB < 16GB).
            # CPU offload is not compatible with 4-bit bitsandbytes weights (which are GPU-only).
            # So we only enable cpu_offload if we are NOT using quantization.
            
            is_quantized = "flux" in model_id_or_path.lower() and "cuda" in self._device
            
            if not is_quantized:
                 # Use CPU offload for non-quantized large models or if we are on low VRAM
                 # For now, we assume non-flux models (like SDXL) might benefit or are small enough.
                 # Actually, standard SD1.5/SDXL fits in 16GB easily too without offload.
                 # But sticking to previous behavior for non-flux is safer? 
                 # Wait, previous behavior was .to(device).
                 # Step 537 changed it to enable_model_cpu_offload unconditionally.
                 # Let's revert to .to(device) for standard models if they fit, or keep offload.
                 # But for Flux 4-bit, we MUST NOT use offload.
                 
                 # Let's use simple logic:
                 # If Flux 4-bit -> .to(device) (implied by from_pretrained behavior, usually already on device)
                 # Actually, from_pretrained with quant_config puts it on device map "auto" or GPU.
                 # We shouldn't move it manually if it's already mapped.
                 pass
            else:
                 # It's quantized Flux. It should stay on GPU.
                 # We might need to ensure it's on the right device if device_map didn't do it.
                 # configuring serialization/offload with bitsandbytes is tricky.
                 # Ideally, we just let it be.
                 pass

            # Update: `enable_model_cpu_offload` is actually very robust in diffusers.
            # But with bnb 4bit, it triggers errors.
            # So we SKIP it for quantized models.
            
            if not is_quantized:
                 self._pipeline.enable_model_cpu_offload(gpu_id=int(settings.sd_gpu_device))
            else:
                 # Ensure pipeline is on the correct device (though components likely already are)
                 # With bitsandbytes, we don't manually .to() the quantized layers.
                 # The pipeline wrapper itself can be moved?
                 # pipeline.to() might fail on quantized layers.
                 # We trust loading placed it correctly.
                 pass

            self._loaded = True
            logger.info("✓ Stable Diffusion model loaded on %s", self._device)

        except Exception:
            logger.exception("Failed to load Image generation model")
            raise

    def unload_model(self) -> None:
        """Unload model to free VRAM."""
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
        
        if "cuda" in self._device:
            torch.cuda.empty_cache()
            
        gc.collect()
        self._loaded = False
        logger.info("Stable Diffusion model unloaded")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> ImageResult:
        """
        Run inference. BLOCKING method (run in executor).
        """
        if not self._loaded or not self._pipeline:
            raise RuntimeError("Image model is not loaded")

        t0 = time.perf_counter()

        # Defaults
        steps = num_steps or settings.sd_default_steps
        guidance = guidance_scale or settings.sd_default_guidance
        w = width or settings.sd_default_width
        h = height or settings.sd_default_height

        # Seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self._device).manual_seed(seed)

        logger.info(
            "Generating image: '%s' (Size: %dx%d, Steps: %d, Seed: %d)",
            prompt, w, h, steps, seed
        )

        if "cuda" in self._device:
            with torch.autocast("cuda"):
                # Prepare args dynamically
                kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "width": w,
                    "height": h,
                    "generator": generator,
                    "output_type": "pil",
                }
                # Flux/Schnell usually doesn't support negative_prompt in strict sense, check or pass safely
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt

                output = self._pipeline(**kwargs)
        else:
             # Prepare args dynamically
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "width": w,
                "height": h,
                "generator": generator,
                "output_type": "pil",
            }
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            output = self._pipeline(**kwargs)

        image: Image.Image = output.images[0]
        
        # Save to cache
        filename = f"img_{int(time.time())}_{seed}.png"
        save_path = Path(settings.image_cache_dir) / filename
        image.save(save_path)

        elapsed = time.perf_counter() - t0
        logger.info("Image generated in %.2fs", elapsed)

        return ImageResult(
            path=save_path,
            seed=seed,
            width=w,
            height=h,
            elapsed_seconds=elapsed,
        )


    def cleanup_cache(self) -> int:
        """
        Cleanup old images from the cache directory.
        Returns:
            int: Number of files deleted.
        """
        if not settings.image_cache_dir:
            return 0
            
        cache_dir = Path(settings.image_cache_dir)
        if not cache_dir.exists():
            return 0
            
        deleted_count = 0
        now = time.time()
        ttl_seconds = int(settings.image_cache_ttl_hours) * 3600
        
        try:
            for file_path in cache_dir.glob("*.png"):
                if file_path.is_file():
                    stat = file_path.stat()
                    if now - stat.st_mtime > ttl_seconds:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug("Deleted old cache file: %s", file_path.name)
        except Exception as e:
            logger.error("Error during cache cleanup: %s", e)
            
        return deleted_count
