"""
Athena — 管理 Cog

オーナー限定スラッシュコマンド: reload, config, shutdown
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands
from discord.ext import commands

from bot.config import settings
from bot.utils.embed_builder import Colour, EMOJI, make_embed, error_embed, success_embed

if TYPE_CHECKING:
    from bot.main import Athena

logger = logging.getLogger("athena.cog.admin")


def _is_owner() -> app_commands.check:
    """デコレーター: コマンドをBotオーナーに限定。"""

    async def predicate(interaction: discord.Interaction) -> bool:
        if settings.owner_id and interaction.user.id == settings.owner_id:
            return True
        # フォールバック: アプリケーションオーナーを確認
        app_info = await interaction.client.application_info()  # type: ignore[union-attr]
        return interaction.user.id == app_info.owner.id

    return app_commands.check(predicate)


class AdminCog(commands.Cog, name="Admin"):
    """オーナー限定管理コマンド。"""

    def __init__(self, bot: Athena) -> None:
        self.bot = bot

    # ── /reload ──────────────────────────────────────────────────────────

    @app_commands.command(name="reload", description="[オーナー] Cogモジュールをホットリロード")
    @app_commands.describe(cog="Cogモジュールパス（例: bot.cogs.chat）")
    @_is_owner()
    async def reload(self, interaction: discord.Interaction, cog: str) -> None:
        await interaction.response.defer(ephemeral=True)

        try:
            await self.bot.reload_extension(cog)
            logger.info("Reloaded cog: %s", cog)
            await interaction.followup.send(
                embed=success_embed(f"`{cog}` のリロードに成功しました。"),
                ephemeral=True,
            )
        except commands.ExtensionNotLoaded:
            await interaction.followup.send(
                embed=error_embed(f"`{cog}` は読み込まれていません。完全なモジュールパスを使用してください。"),
                ephemeral=True,
            )
        except commands.ExtensionNotFound:
            await interaction.followup.send(
                embed=error_embed(f"`{cog}` が見つかりません。"),
                ephemeral=True,
            )
        except Exception as exc:
            logger.exception("Failed to reload %s", cog)
            await interaction.followup.send(
                embed=error_embed(f"`{cog}` のリロードに失敗: {exc}"),
                ephemeral=True,
            )

    # ── /config ──────────────────────────────────────────────────────────

    @app_commands.command(name="config", description="[オーナー] 設定の閲覧・変更")
    @app_commands.describe(
        key="設定名（空欄で全設定を表示）",
        value="新しい値（空欄で現在の値を表示）",
    )
    @_is_owner()
    async def config(
        self,
        interaction: discord.Interaction,
        key: Optional[str] = None,
        value: Optional[str] = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)

        if key is None:
            # 全設定を表示（トークンはマスク）
            pairs: list[str] = []
            for field_name in settings.model_fields:
                val = getattr(settings, field_name, "?")
                if "token" in field_name.lower() or "secret" in field_name.lower():
                    val = "••••••••"
                pairs.append(f"`{field_name}` = `{val}`")

            text = "\n".join(pairs)
            if len(text) > 3900:
                text = text[:3900] + "\n\n*… 省略*"

            embed = make_embed(
                "現在の設定",
                text,
                colour=Colour.INFO,
                emoji=EMOJI["gear"],
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
            return

        # 特定の設定を閲覧・変更
        if not hasattr(settings, key):
            await interaction.followup.send(
                embed=error_embed(f"不明な設定: `{key}`"),
                ephemeral=True,
            )
            return

        if value is None:
            # 閲覧
            current = getattr(settings, key)
            if "token" in key.lower() or "secret" in key.lower():
                current = "••••••••"
            await interaction.followup.send(
                embed=make_embed(
                    "設定値",
                    f"`{key}` = `{current}`",
                    colour=Colour.INFO,
                    emoji=EMOJI["gear"],
                ),
                ephemeral=True,
            )
            return

        # 更新（ベストエフォートの型変換）
        field_info = settings.model_fields.get(key)
        if field_info is None:
            await interaction.followup.send(
                embed=error_embed(f"`{key}` は更新できません。"),
                ephemeral=True,
            )
            return

        try:
            annotation = field_info.annotation
            if annotation is int:
                coerced = int(value)
            elif annotation is float:
                coerced = float(value)
            elif annotation is bool:
                coerced = value.lower() in ("true", "1", "yes")
            else:
                coerced = value

            setattr(settings, key, coerced)
            logger.info("Config updated: %s = %s", key, coerced)

            await interaction.followup.send(
                embed=success_embed(f"`{key}` を `{coerced}` に更新しました"),
                ephemeral=True,
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=error_embed(f"`{key}` の設定に失敗: {exc}"),
                ephemeral=True,
            )

    # ── /shutdown ────────────────────────────────────────────────────────

    @app_commands.command(name="shutdown", description="[オーナー] Botを安全にシャットダウン")
    @_is_owner()
    async def shutdown(self, interaction: discord.Interaction) -> None:
        embed = make_embed(
            "シャットダウン中",
            "Athena を安全にシャットダウンしています。さようなら！👋",
            colour=Colour.WARNING,
            emoji=EMOJI["warning"],
        )
        await interaction.response.send_message(embed=embed)
        logger.info("Shutdown requested by %s", interaction.user)
        await self.bot.close()

    # ── エラーハンドラ ───────────────────────────────────────────────────

    async def cog_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message(
                embed=error_embed("このコマンドを使用する権限がありません。"),
                ephemeral=True,
            )
        else:
            logger.exception("Admin command error: %s", error)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    embed=error_embed(f"エラーが発生しました: {error}"),
                    ephemeral=True,
                )


async def setup(bot: Athena) -> None:
    await bot.add_cog(AdminCog(bot))
