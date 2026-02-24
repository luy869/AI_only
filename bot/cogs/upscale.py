"""
Athena — Upscale Cog

Real-ESRGAN を使って画像を拡大するスラッシュコマンド。
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands
from PIL import Image

from bot.utils.embed_builder import (
    Colour,
    EMOJI,
    make_embed,
    error_embed,
)
from bot.utils.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from bot.main import Athena

logger = logging.getLogger("athena.cog.upscale")

_limiter = RateLimiter()


class UpscaleCog(commands.Cog, name="Upscale"):
    """画像高画質化（超解像）コマンド。"""

    def __init__(self, bot: Athena) -> None:
        self.bot = bot

    # ── /upscale ─────────────────────────────────────────────────────────

    @app_commands.command(
        name="upscale",
        description="Real-ESRGAN を用いて画像を高品質に拡大します",
    )
    @app_commands.describe(
        image="拡大したい画像ファイル",
        scale="拡大倍率（2倍または4倍）※デフォルトは2倍",
    )
    @app_commands.choices(
        scale=[
            app_commands.Choice(name="2x (安全・高速)", value=2.0),
            app_commands.Choice(name="4x (高負荷・高画質)", value=4.0),
        ]
    )
    async def upscale(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        scale: app_commands.Choice[float] = None, # type: ignore
    ) -> None:
        # レートリミット
        if not _limiter.try_acquire(interaction.user.id):
            retry = _limiter.retry_after(interaction.user.id)
            await interaction.response.send_message(
                embed=error_embed(f"レート制限中です。{retry:.0f}秒後に再試行してください。"),
                ephemeral=True,
            )
            return

        # サービス準備チェック
        if self.bot.image_service is None:
            await interaction.response.send_message(
                embed=error_embed("画像モデルを未ロードまたは準備中です。"),
                ephemeral=True,
            )
            return
        if self.bot.queue_service is None:
            await interaction.response.send_message(
                embed=error_embed("キューサービスが利用できません。"),
                ephemeral=True,
            )
            return

        # 画像形式とサイズのチェック
        if not image.content_type or not image.content_type.startswith("image/"):
            await interaction.response.send_message(
                embed=error_embed("有効な画像ファイルを添付してください。"),
                ephemeral=True,
            )
            return

        # メモリ制限のための安全基準（短辺1536px以下）
        # ただしAttachmentオブジェクト自体からは即座にheight/widthを取れない場合があるため推測
        if image.width and image.height:
            short_side = min(image.width, image.height)
            if short_side > 1536:
                await interaction.response.send_message(
                    embed=error_embed(f"画像が大きすぎます (短辺 {short_side}px)。メモリ爆発を防ぐため、元の画像の短辺は1536px以下にしてください。"),
                    ephemeral=True,
                )
                return

        # 倍率
        outscale = scale.value if scale else 2.0

        # Defer — アップスケールもCPU処理のため数十秒かかる想定
        await interaction.response.defer(thinking=True)

        try:
            # 画像をメモリにダウンロード
            img_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # 再度サイズチェック (念のため)
            short_side = min(pil_image.width, pil_image.height)
            if short_side > 1536:
                 await interaction.followup.send(
                    embed=error_embed(f"画像が大きすぎます (短辺 {short_side}px)。メモリ爆発を防ぐため1536px以下の画像を使用してください。")
                 )
                 return

            # 画像キューにアップスケール処理を投入（Flux生成と並行稼働によるOOMを防ぐため直列）
            result = await self.bot.queue_service.submit_image(
                self.bot.image_service.upscale,
                pil_image,
                outscale=outscale,
            )

            # 生成画像のリッチ Embed を構築
            file = discord.File(str(result.path), filename=result.path.name)

            embed = make_embed(
                "画像拡大 (Upscale) 完了",
                f"**ファイルを拡大しました**",
                colour=Colour.IMAGE,
                emoji=EMOJI["imagine"],
                fields=[
                    ("倍率", f"{outscale}x", True),
                    ("元のサイズ", f"{pil_image.width}×{pil_image.height}", True),
                    ("拡大後サイズ", f"{result.width}×{result.height}", True),
                    ("処理時間", f"{result.elapsed_seconds:.1f}秒 (CPU)", True),
                ],
            )
            embed.set_image(url=f"attachment://{result.path.name}")
            embed.set_footer(
                text=f"{interaction.user.display_name} のリクエスト • Athena Upscaler",
                icon_url=(
                    interaction.user.display_avatar.url
                    if interaction.user.display_avatar
                    else None
                ),
            )

            await interaction.followup.send(embed=embed, file=file)

        except asyncio.QueueFull:
            await interaction.followup.send(
                embed=error_embed("画像処理キューが満杯です。しばらく経ってから再試行してください。")
            )
        except Exception as e:
            logger.exception("Upscale command failed")
            await interaction.followup.send(
                embed=error_embed(f"画像拡大に失敗しました。詳細: {e}")
            )


async def setup(bot: Athena) -> None:
    await bot.add_cog(UpscaleCog(bot))
