"""
Athena — チャット Cog

AI会話、翻訳、要約、コード支援のスラッシュコマンド。
すべての重い推論はキューサービス経由でディスパッチ。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from bot.utils.embed_builder import (
    Colour,
    EMOJI,
    make_embed,
    error_embed,
    loading_embed,
)
from bot.utils.rate_limiter import RateLimiter
from bot.prompts import CODE_SYSTEM, SUMMARIZE_SYSTEM, TRANSLATE_SYSTEM
from bot.services.llm_service import (
    LLMError,
    LLMModelNotLoaded,
    LLMAPIError,
    LLMTimeoutError,
    LLMConnectionError,
)

if TYPE_CHECKING:
    from bot.main import Athena

logger = logging.getLogger("athena.cog.chat")

# チャットコマンド共通のレートリミッター
_limiter = RateLimiter()

# ── 特殊コマンド用システムプロンプト ──────────────────────────────────────
# Imported from bot.prompts


class ChatCog(commands.Cog, name="Chat"):
    """AI会話コマンド。"""

    def __init__(self, bot: Athena) -> None:
        self.bot = bot

    def _check_services(self) -> str | None:
        """サービスが準備できていない場合はエラーメッセージを返す。"""
        if self.bot.llm_service is None or not self.bot.llm_service.is_loaded:
            return "LLMモデルをまだ読み込み中です。しばらくお待ちください。"
        if self.bot.queue_service is None:
            return "キューサービスが利用できません。"
        return None

    # ── /chat ────────────────────────────────────────────────────────────

    @app_commands.command(name="chat", description="Athena と会話 — 文脈を記憶するAIチャット")
    @app_commands.describe(message="Athena へのメッセージ")
    async def chat(self, interaction: discord.Interaction, message: str) -> None:
        # レートリミット
        if not _limiter.try_acquire(interaction.user.id):
            retry = _limiter.retry_after(interaction.user.id)
            await interaction.response.send_message(
                embed=error_embed(f"レート制限中です。{retry:.0f}秒後に再試行してください。"),
                ephemeral=True,
            )
            return

        # サービス準備チェック
        err = self._check_services()
        if err:
            await interaction.response.send_message(embed=error_embed(err), ephemeral=True)
            return

        # Defer — 推論に時間がかかる
        await interaction.response.defer(thinking=True)

        try:
            assert self.bot.llm_service is not None
            assert self.bot.queue_service is not None

            response = await self.bot.queue_service.submit_llm(
                self.bot.llm_service.generate,
                interaction.user.id,
                message,
            )

            # 必要に応じて切り詰め（Discord制限 = Embed descriptionは4096文字）
            if len(response) > 3900:
                response = response[:3900] + "\n\n*… (省略)*"

            embed = make_embed(
                "チャット",
                response,
                colour=Colour.PRIMARY,
                emoji=EMOJI["chat"],
                author_name=interaction.user.display_name,
                author_icon_url=(
                    interaction.user.display_avatar.url
                    if interaction.user.display_avatar
                    else None
                ),
            )
            await interaction.followup.send(embed=embed)

        except asyncio.QueueFull:
            await interaction.followup.send(
                embed=error_embed("キューが満杯です！しばらく経ってから再試行してください。")
            )
        except LLMModelNotLoaded:
            await interaction.followup.send(
                embed=loading_embed("モデルを読み込み中です。しばらくお待ちください...")
            )
        except (LLMTimeoutError, LLMConnectionError):
            await interaction.followup.send(
                embed=error_embed("LLMサーバーとの通信に失敗しました。しばらく待ってから再試行してください。")
            )
        except LLMAPIError as e:
            await interaction.followup.send(
                embed=error_embed(f"LLMエラーが発生しました: {e.message}")
            )
        except Exception:
            logger.exception("Chat command failed")
            await interaction.followup.send(
                embed=error_embed("予期しないエラーが発生しました。")
            )

    # ── /clear ───────────────────────────────────────────────────────────

    @app_commands.command(name="clear", description="Athena との会話履歴をリセット")
    async def clear(self, interaction: discord.Interaction) -> None:
        if self.bot.llm_service:
            self.bot.llm_service.clear_history(interaction.user.id)

        embed = make_embed(
            "履歴クリア",
            "会話履歴がリセットされました。",
            colour=Colour.SUCCESS,
            emoji=EMOJI["success"],
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ── /translate ───────────────────────────────────────────────────────

    @app_commands.command(name="translate", description="テキストを別の言語に翻訳")
    @app_commands.describe(
        text="翻訳するテキスト",
        target_language="翻訳先の言語（デフォルト: 日本語）",
    )
    async def translate(
        self,
        interaction: discord.Interaction,
        text: str,
        target_language: str = "日本語",
    ) -> None:
        if not _limiter.try_acquire(interaction.user.id):
            retry = _limiter.retry_after(interaction.user.id)
            await interaction.response.send_message(
                embed=error_embed(f"レート制限中です。{retry:.0f}秒後に再試行してください。"),
                ephemeral=True,
            )
            return

        err = self._check_services()
        if err:
            await interaction.response.send_message(embed=error_embed(err), ephemeral=True)
            return

        await interaction.response.defer(thinking=True)

        prompt = f"以下のテキストを{target_language}に翻訳してください:\n\n{text}"

        try:
            assert self.bot.llm_service is not None
            assert self.bot.queue_service is not None

            result = await self.bot.queue_service.submit_llm(
                self.bot.llm_service.generate,
                interaction.user.id,
                prompt,
                system_override=TRANSLATE_SYSTEM,
            )

            embed = make_embed(
                f"翻訳 → {target_language}",
                result,
                colour=Colour.INFO,
                emoji=EMOJI["translate"],
                fields=[("原文", text[:1024], False)],
            )
            await interaction.followup.send(embed=embed)

        except asyncio.QueueFull:
            await interaction.followup.send(
                embed=error_embed("キューが満杯です。しばらく経ってから再試行してください。")
            )
        except LLMModelNotLoaded:
            await interaction.followup.send(
                embed=loading_embed("モデルを読み込み中です。しばらくお待ちください...")
            )
        except (LLMTimeoutError, LLMConnectionError):
            await interaction.followup.send(
                embed=error_embed("LLMサーバーとの通信に失敗しました。")
            )
        except LLMAPIError as e:
            await interaction.followup.send(
                embed=error_embed(f"翻訳エラー: {e.message}")
            )
        except Exception:
            logger.exception("Translate command failed")
            await interaction.followup.send(
                embed=error_embed("翻訳に失敗しました。もう一度お試しください。")
            )

    # ── /summarize ───────────────────────────────────────────────────────

    @app_commands.command(name="summarize", description="長いテキストを要約")
    @app_commands.describe(text="要約するテキスト")
    async def summarize(self, interaction: discord.Interaction, text: str) -> None:
        if not _limiter.try_acquire(interaction.user.id):
            retry = _limiter.retry_after(interaction.user.id)
            await interaction.response.send_message(
                embed=error_embed(f"レート制限中です。{retry:.0f}秒後に再試行してください。"),
                ephemeral=True,
            )
            return

        err = self._check_services()
        if err:
            await interaction.response.send_message(embed=error_embed(err), ephemeral=True)
            return

        await interaction.response.defer(thinking=True)

        try:
            assert self.bot.llm_service is not None
            assert self.bot.queue_service is not None

            result = await self.bot.queue_service.submit_llm(
                self.bot.llm_service.generate,
                interaction.user.id,
                text,
                system_override=SUMMARIZE_SYSTEM,
            )

            embed = make_embed(
                "要約",
                result,
                colour=Colour.INFO,
                emoji=EMOJI["summarize"],
            )
            await interaction.followup.send(embed=embed)

        except asyncio.QueueFull:
            await interaction.followup.send(
                embed=error_embed("キューが満杯です。しばらく経ってから再試行してください。")
            )
        except LLMModelNotLoaded:
            await interaction.followup.send(
                embed=loading_embed("モデルを読み込み中です。しばらくお待ちください...")
            )
        except (LLMTimeoutError, LLMConnectionError):
            await interaction.followup.send(
                embed=error_embed("LLMサーバーとの通信に失敗しました。")
            )
        except LLMAPIError as e:
            await interaction.followup.send(
                embed=error_embed(f"要約エラー: {e.message}")
            )
        except Exception:
            logger.exception("Summarize command failed")
            await interaction.followup.send(
                embed=error_embed("要約に失敗しました。もう一度お試しください。")
            )

    # ── /code ────────────────────────────────────────────────────────────

    @app_commands.command(name="code", description="コードの支援 — レビュー・生成・解説")
    @app_commands.describe(request="依頼内容を記述してください")
    async def code(self, interaction: discord.Interaction, request: str) -> None:
        if not _limiter.try_acquire(interaction.user.id):
            retry = _limiter.retry_after(interaction.user.id)
            await interaction.response.send_message(
                embed=error_embed(f"レート制限中です。{retry:.0f}秒後に再試行してください。"),
                ephemeral=True,
            )
            return

        err = self._check_services()
        if err:
            await interaction.response.send_message(embed=error_embed(err), ephemeral=True)
            return

        await interaction.response.defer(thinking=True)

        try:
            assert self.bot.llm_service is not None
            assert self.bot.queue_service is not None

            result = await self.bot.queue_service.submit_llm(
                self.bot.llm_service.generate,
                interaction.user.id,
                request,
                system_override=CODE_SYSTEM,
            )

            # コードの回答は長くなりがち — 必要に応じて切り詰め
            if len(result) > 3900:
                result = result[:3900] + "\n\n*… (省略)*"

            embed = make_embed(
                "コードアシスタント",
                result,
                colour=Colour.PRIMARY,
                emoji=EMOJI["code"],
            )
            await interaction.followup.send(embed=embed)

        except asyncio.QueueFull:
            await interaction.followup.send(
                embed=error_embed("キューが満杯です。しばらく経ってから再試行してください。")
            )
        except LLMModelNotLoaded:
            await interaction.followup.send(
                embed=loading_embed("モデルを読み込み中です。しばらくお待ちください...")
            )
        except (LLMTimeoutError, LLMConnectionError):
            await interaction.followup.send(
                embed=error_embed("LLMサーバーとの通信に失敗しました。")
            )
        except LLMAPIError as e:
            await interaction.followup.send(
                embed=error_embed(f"コード生成エラー: {e.message}")
            )
        except Exception:
            logger.exception("Code command failed")
            await interaction.followup.send(
                embed=error_embed("コード支援に失敗しました。もう一度お試しください。")
            )


async def setup(bot: Athena) -> None:
    await bot.add_cog(ChatCog(bot))
