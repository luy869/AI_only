"""
Athena — 画像生成 Cog

FLUX.1-schnell を使ったテキストから画像生成のスラッシュコマンド。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands
from discord.ext import commands

from bot.utils.embed_builder import (
    Colour,
    EMOJI,
    make_embed,
    error_embed,
)
from bot.utils.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from bot.main import Athena

logger = logging.getLogger("athena.cog.imagine")

_limiter = RateLimiter()


class ImagineCog(commands.Cog, name="Imagine"):
    """画像生成コマンド。"""

    def __init__(self, bot: Athena) -> None:
        self.bot = bot

    # ── /imagine ─────────────────────────────────────────────────────────

    @app_commands.command(
        name="imagine",
        description="テキストから画像を生成",
    )
    @app_commands.describe(
        prompt="生成したい内容（詳しく書くほど良い結果に！）",
        negative_prompt="画像に含めたくない要素（※Fluxでは効果なし）",
        steps="推論ステップ数（1〜50、Schnellデフォルト4）",
        guidance="ガイダンススケール（0〜20、Schnellデフォルト0.0）",
        width="画像の幅 (px)（デフォルト1024）",
        height="画像の高さ (px)（デフォルト1024）",
        seed="再現用シード（省略するとランダム）",
    )
    async def imagine(
        self,
        interaction: discord.Interaction,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[app_commands.Range[int, 1, 50]] = None,
        guidance: Optional[app_commands.Range[float, 0.0, 20.0]] = None,
        width: Optional[app_commands.Range[int, 256, 2048]] = None,
        height: Optional[app_commands.Range[int, 256, 2048]] = None,
        seed: Optional[int] = None,
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
        if self.bot.image_service is None or not self.bot.image_service.is_loaded:
            await interaction.response.send_message(
                embed=error_embed("画像モデルをまだ読み込み中です。しばらくお待ちください。"),
                ephemeral=True,
            )
            return
        if self.bot.queue_service is None:
            await interaction.response.send_message(
                embed=error_embed("キューサービスが利用できません。"),
                ephemeral=True,
            )
            return

        # Defer — 画像生成は10~30秒かかる
        await interaction.response.defer(thinking=True)

        try:
            result = await self.bot.queue_service.submit_image(
                self.bot.image_service.generate,
                prompt,
                negative_prompt=negative_prompt or "",
                num_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                seed=seed,
            )

            # 生成画像のリッチ Embed を構築
            file = discord.File(str(result.path), filename="generated.png")

            embed = make_embed(
                "画像生成完了",
                f"**プロンプト:** {prompt}",
                colour=Colour.IMAGE,
                emoji=EMOJI["imagine"],
                fields=[
                    ("シード", str(result.seed), True),
                    ("サイズ", f"{result.width}×{result.height}", True),
                    ("生成時間", f"{result.elapsed_seconds:.1f}秒", True),
                ],
            )

            if negative_prompt:
                embed.add_field(
                    name="ネガティブプロンプト",
                    value=negative_prompt[:1024],
                    inline=False,
                )

            embed.set_image(url="attachment://generated.png")
            embed.set_footer(
                text=f"{interaction.user.display_name} のリクエスト • Athena",
                icon_url=(
                    interaction.user.display_avatar.url
                    if interaction.user.display_avatar
                    else None
                ),
            )

            await interaction.followup.send(embed=embed, file=file)

        except asyncio.QueueFull:
            await interaction.followup.send(
                embed=error_embed(
                    "画像生成キューが満杯です。しばらく経ってから再試行してください。"
                )
            )
        except ValueError as exc:
            # NSFW コンテンツブロック
            await interaction.followup.send(
                embed=error_embed(str(exc)),
            )
        except Exception:
            logger.exception("Imagine command failed")
            await interaction.followup.send(
                embed=error_embed("画像生成に失敗しました。もう一度お試しください。")
            )


async def setup(bot: Athena) -> None:
    await bot.add_cog(ImagineCog(bot))
