"""
Athena — Image Generation Service

FLUX.1-schnell による高速テキスト→画像生成。
4-bit 量子化 (bitsandbytes) で VRAM 使用量を最適化。
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
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
    """Manages Flux image generation model and inference."""

    def __init__(self) -> None:
        self._pipeline = None
        self._loaded = False
        self._device = (
            f"cuda:{settings.sd_gpu_device}"
            if torch.cuda.is_available()
            else "cpu"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Model Loading ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load FLUX model into VRAM with 4-bit quantization."""
        if self._loaded:
            return

        model_id = settings.sd_model_id
        logger.info("Loading image model: %s ...", model_id)

        try:
            # RTX 5080 (Blackwell) / Ampere は bfloat16 が最適
            # float16 は LayerNorm 等で dtype mismatch 警告の原因になる
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability(int(settings.sd_gpu_device))
                dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
            else:
                dtype = torch.float32

            if "flux" in model_id.lower() and "cuda" in self._device:
                self._load_flux_quantized(model_id, dtype)
            else:
                self._load_standard(model_id, dtype)

            self._loaded = True
            logger.info("✓ Image model loaded on %s", self._device)

        except Exception:
            logger.exception("Failed to load image generation model")
            raise

    def _load_flux_quantized(self, model_id: str, dtype: torch.dtype) -> None:
        """Load Flux model with 4-bit quantization for VRAM efficiency."""
        from transformers import BitsAndBytesConfig, T5EncoderModel
        from diffusers import FluxTransformer2DModel, AutoPipelineForText2Image

        gpu_id = int(settings.sd_gpu_device)
        logger.info("  Flux detected — enabling 4-bit quantization on cuda:%d ...", gpu_id)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

        # Transformer (4-bit → GPU)
        logger.info("  Loading transformer (4-bit) ...")
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map=f"cuda:{gpu_id}",
            cache_dir=settings.sd_model_local_path,
        )

        # T5 Text Encoder (4-bit → GPU — saves ~6GB)
        logger.info("  Loading text encoder (4-bit) ...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map=f"cuda:{gpu_id}",
            cache_dir=settings.sd_model_local_path,
        )

        # Assemble pipeline — remaining components (VAE, CLIP, scheduler)
        # are loaded normally and moved to GPU explicitly
        logger.info("  Assembling pipeline ...")
        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=dtype,
            cache_dir=settings.sd_model_local_path,
        )

        # ── GPU/CPU 配置方針 ────────────────────────────────────────────────
        # まず pipeline 全体を GPU に移動することで pipeline.device = cuda:0 を確立する。
        # (これをしないと diffusers 内部のデバイス判定が cpu になり T5 で OOM が起きる)
        # その後で VRAM 節約のため VAE だけ CPU に戻す。
        device = torch.device(f"cuda:{gpu_id}")

        logger.info("  Moving pipeline to GPU (%s) ...", device)
        self._pipeline = self._pipeline.to(device)

        # VAE → CPU（デコード時のVRAM OOMを根本的に防ぐ）
        if hasattr(self._pipeline, 'vae') and self._pipeline.vae is not None:
            self._pipeline.vae = self._pipeline.vae.to("cpu", dtype=dtype)
            self._pipeline.vae.enable_tiling()
            self._pipeline.vae.enable_slicing()
            logger.info("  VAE → CPU (tiling + slicing enabled)")

            # latents は CUDA 上で生成されるため、CPU VAE に渡す前に変換するラッパー
            original_decode = self._pipeline.vae.decode

            def decode_wrapper(latents, *args, **kwargs):
                latents = latents.to("cpu")
                logger.info("  Decoding latents on CPU...")
                torch.cuda.empty_cache()
                return original_decode(latents, *args, **kwargs)

            self._pipeline.vae.decode = decode_wrapper


        # VRAM 残量を確認してログ出力
        torch.cuda.empty_cache()
        vram_free = (
            torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
            if torch.cuda.is_available() else 0
        )
        logger.info("  VRAM free after setup: %.2f GB", vram_free)

        logger.info("  ✓ Quantized Flux pipeline ready (cuda:%d, dtype=%s)", gpu_id, dtype)


    def _load_standard(self, model_id: str, dtype: torch.dtype) -> None:
        """Load a standard diffusers model (SDXL, SD v1.5, etc.)."""
        from diffusers import AutoPipelineForText2Image

        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=settings.sd_model_local_path,
        )

        # CPU offload for non-quantized models
        self._pipeline.enable_model_cpu_offload(
            gpu_id=int(settings.sd_gpu_device)
        )

    # ── Model Unloading ──────────────────────────────────────────────────

    def unload_model(self) -> None:
        """Unload model to free VRAM."""
        if self._pipeline:
            del self._pipeline
            self._pipeline = None

        if "cuda" in self._device:
            torch.cuda.empty_cache()

        gc.collect()
        self._loaded = False
        logger.info("Image model unloaded")

    # ── Inference ─────────────────────────────────────────────────────────

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
        Run inference.  BLOCKING method — call via run_in_executor.
        """
        if not self._loaded or not self._pipeline:
            raise RuntimeError("Image model is not loaded")

        t0 = time.perf_counter()

        # Defaults
        steps = num_steps or settings.sd_default_steps
        guidance = guidance_scale if guidance_scale is not None else settings.sd_default_guidance
        w = width or settings.sd_default_width
        h = height or settings.sd_default_height

        # Seed — use CPU generator (required for bitsandbytes quantized models)
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            "Generating image: '%s' (Size: %dx%d, Steps: %d, Seed: %d)",
            prompt, w, h, steps, seed,
        )

        # Build kwargs
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": w,
            "height": h,
            "generator": generator,
            "output_type": "pil",
        }

        # Flux does not support negative_prompt — only add for non-Flux models
        is_flux = "flux" in settings.sd_model_id.lower()
        if negative_prompt and not is_flux:
            kwargs["negative_prompt"] = negative_prompt

        # 推論実行
        # autocast は bitsandbytes 量子化と非互換のため使用しない
        # 代わりに推論前後でキャッシュをクリアして VRAM フラグメントを防ぐ
        torch.cuda.empty_cache()
        output = self._pipeline(**kwargs)

        image: Image.Image = output.images[0]

        # Save to cache
        filename = f"img_{int(time.time())}_{seed}.png"
        save_path = Path(settings.image_cache_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
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

    # ── Cache Management ──────────────────────────────────────────────────

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
