"""
Athena — ユーティリティ Cog

情報系スラッシュコマンド: help, ping, stats
"""

from __future__ import annotations

import logging
import platform
import time
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands, tasks

from bot.utils.embed_builder import Colour, EMOJI, make_embed

if TYPE_CHECKING:
    from bot.main import Athena

logger = logging.getLogger("athena.cog.utility")


class UtilityCog(commands.Cog, name="Utility"):
    """情報系コマンドとバックグラウンドタスク。"""

    def __init__(self, bot: Athena) -> None:
        self.bot = bot
        self._start_time = time.time()
        self.cache_cleanup_loop.start()

    def cog_unload(self) -> None:
        self.cache_cleanup_loop.cancel()

    # ── バックグラウンドタスク ────────────────────────────────────────────

    @tasks.loop(hours=1)
    async def cache_cleanup_loop(self) -> None:
        """古い生成画像を定期的にクリーンアップ。"""
        if self.bot.image_service:
            deleted = self.bot.image_service.cleanup_cache()
            if deleted:
                logger.info("定期キャッシュ清掃: %d 枚の画像を削除", deleted)

    @cache_cleanup_loop.before_loop
    async def _before_cleanup(self) -> None:
        await self.bot.wait_until_ready()

    # ── /help ────────────────────────────────────────────────────────────

    @app_commands.command(name="help", description="全コマンドの一覧を表示")
    async def help_cmd(self, interaction: discord.Interaction) -> None:
        embed = make_embed(
            "Athena — コマンドガイド",
            "ローカルGPUで動作するセルフホストAIアシスタント",
            colour=Colour.PRIMARY,
            emoji=EMOJI["help"],
        )

        # 会話コマンド
        embed.add_field(
            name=f"{EMOJI['chat']}  会話",
            value=(
                "`/chat` — Athena と会話（文脈を記憶）\n"
                "`/clear` — 会話履歴をリセット\n"
            ),
            inline=False,
        )

        # AIツール
        embed.add_field(
            name=f"{EMOJI['code']}  AI ツール",
            value=(
                "`/code` — コードレビュー・生成・解説\n"
                "`/translate` — テキストを任意の言語に翻訳\n"
                "`/summarize` — 長いテキストを要約\n"
            ),
            inline=False,
        )

        # 画像生成
        embed.add_field(
            name=f"{EMOJI['imagine']}  画像生成",
            value=(
                "`/imagine` — テキストから画像を生成\n"
                "  オプション: `prompt`, `negative_prompt`, `steps`, "
                "`guidance`, `width`, `height`, `seed`\n"
            ),
            inline=False,
        )

        # 情報
        embed.add_field(
            name=f"{EMOJI['stats']}  情報",
            value=(
                "`/ping` — レイテンシを確認\n"
                "`/stats` — GPU / キュー / メモリ統計\n"
                "`/help` — このメッセージ\n"
            ),
            inline=False,
        )

        await interaction.response.send_message(embed=embed)

    # ── /ping ────────────────────────────────────────────────────────────

    @app_commands.command(name="ping", description="Botのレイテンシを確認")
    async def ping(self, interaction: discord.Interaction) -> None:
        ws_latency = self.bot.latency * 1000  # ms

        t0 = time.perf_counter()
        await interaction.response.defer(thinking=True)
        api_latency = (time.perf_counter() - t0) * 1000

        embed = make_embed(
            "ポン！",
            "",
            colour=Colour.SUCCESS,
            emoji=EMOJI["ping"],
            fields=[
                ("WebSocket", f"`{ws_latency:.0f}ms`", True),
                ("API", f"`{api_latency:.0f}ms`", True),
            ],
        )
        await interaction.followup.send(embed=embed)

    # ── /stats ───────────────────────────────────────────────────────────

    @app_commands.command(name="stats", description="システム・Bot統計を表示")
    async def stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)

        # 稼働時間
        uptime_secs = time.time() - self._start_time
        hours, rem = divmod(int(uptime_secs), 3600)
        minutes, secs = divmod(rem, 60)
        uptime_str = f"{hours}時間 {minutes}分 {secs}秒"

        embed = make_embed(
            "システム統計",
            "",
            colour=Colour.INFO,
            emoji=EMOJI["stats"],
        )

        # Bot情報
        embed.add_field(
            name="🤖 Bot",
            value=(
                f"**稼働時間:** {uptime_str}\n"
                f"**サーバー数:** {len(self.bot.guilds)}\n"
                f"**Python:** {platform.python_version()}\n"
                f"**discord.py:** {discord.__version__}\n"
            ),
            inline=True,
        )

        # GPU情報
        gpu_info = self._get_gpu_info()
        embed.add_field(
            name="🖥️ GPU",
            value=gpu_info,
            inline=True,
        )

        # キュー統計
        if self.bot.queue_service:
            qs = self.bot.queue_service.stats
            embed.add_field(
                name="📋 キュー",
                value=(
                    f"**LLM キュー:** {qs['llm_queue_size']}/{qs['llm_queue_max']}\n"
                    f"**LLM 処理数:** {qs['llm_processed']}\n"
                    f"**LLM 平均時間:** {qs['llm_avg_time']:.1f}秒\n"
                    f"**画像 キュー:** {qs['image_queue_size']}/{qs['image_queue_max']}\n"
                    f"**画像 処理数:** {qs['image_processed']}\n"
                    f"**画像 平均時間:** {qs['image_avg_time']:.1f}秒\n"
                ),
                inline=False,
            )

        # モデル状態
        llm_status = "✅ ロード済み" if (self.bot.llm_service and self.bot.llm_service.is_loaded) else "⏳ 読み込み中"
        sd_status = "✅ ロード済み" if (self.bot.image_service and self.bot.image_service.is_loaded) else "⏳ 読み込み中"

        embed.add_field(
            name="🧠 モデル",
            value=(
                f"**LLM:** {llm_status}\n"
                f"**Stable Diffusion:** {sd_status}\n"
            ),
            inline=True,
        )

        await interaction.followup.send(embed=embed)

    @staticmethod
    def _get_gpu_info() -> str:
        """nvidia-smi で GPU 使用率を取得。"""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                parts = []
                for i, line in enumerate(lines):
                    cols = [c.strip() for c in line.split(",")]
                    if len(cols) == 4:
                        name, util, mem_used, mem_total = cols
                        parts.append(
                            f"**GPU {i}:** {name}\n"
                            f"  使用率: {util}% │ VRAM: {mem_used}/{mem_total} MB"
                        )
                return "\n".join(parts) if parts else "N/A"
        except Exception:
            pass
        return "nvidia-smi 利用不可"


async def setup(bot: Athena) -> None:
    await bot.add_cog(UtilityCog(bot))
