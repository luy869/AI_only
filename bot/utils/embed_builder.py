"""
Athena — Discord Embed Builder

統一されたブランドカラー・フッター・タイムスタンプで
一貫した見た目の Embed を生成するユーティリティ。
"""

from __future__ import annotations

import datetime
from typing import Optional

import discord


# ── カラーパレット ────────────────────────────────────────────────────────
class Colour:
    """Athena ブランドカラー"""

    PRIMARY = discord.Colour.from_rgb(99, 102, 241)   # Indigo-500
    SUCCESS = discord.Colour.from_rgb(34, 197, 94)     # Green-500
    WARNING = discord.Colour.from_rgb(234, 179, 8)     # Yellow-500
    ERROR = discord.Colour.from_rgb(239, 68, 68)       # Red-500
    INFO = discord.Colour.from_rgb(59, 130, 246)       # Blue-500
    IMAGE = discord.Colour.from_rgb(168, 85, 247)      # Purple-500


# ── 絵文字定数 ───────────────────────────────────────────────────────────
EMOJI = {
    "chat": "💬",
    "imagine": "🎨",
    "code": "💻",
    "translate": "🌐",
    "summarize": "📝",
    "stats": "📊",
    "ping": "🏓",
    "help": "📖",
    "warning": "⚠️",
    "error": "❌",
    "success": "✅",
    "loading": "⏳",
    "sparkle": "✦",
    "gear": "⚙️",
}


def make_embed(
    title: str,
    description: str = "",
    *,
    colour: discord.Colour = Colour.PRIMARY,
    emoji: Optional[str] = None,
    footer_text: str = "Athena • ローカル AI アシスタント",
    timestamp: bool = True,
    thumbnail_url: Optional[str] = None,
    image_url: Optional[str] = None,
    author_name: Optional[str] = None,
    author_icon_url: Optional[str] = None,
    fields: Optional[list[tuple[str, str, bool]]] = None,
) -> discord.Embed:
    """統一スタイルの Discord Embed を作成する。"""
    if emoji:
        title = f"{emoji}  {title}"

    embed = discord.Embed(
        title=title,
        description=description,
        colour=colour,
    )

    if timestamp:
        embed.timestamp = datetime.datetime.now(datetime.timezone.utc)

    if footer_text:
        embed.set_footer(text=footer_text)

    if thumbnail_url:
        embed.set_thumbnail(url=thumbnail_url)

    if image_url:
        embed.set_image(url=image_url)

    if author_name:
        embed.set_author(name=author_name, icon_url=author_icon_url or "")

    if fields:
        for name, value, inline in fields:
            embed.add_field(name=name, value=value, inline=inline)

    return embed


def error_embed(message: str, *, title: str = "エラー") -> discord.Embed:
    """エラー用 Embed のショートカット。"""
    return make_embed(title, message, colour=Colour.ERROR, emoji=EMOJI["error"])


def success_embed(message: str, *, title: str = "成功") -> discord.Embed:
    """成功用 Embed のショートカット。"""
    return make_embed(title, message, colour=Colour.SUCCESS, emoji=EMOJI["success"])


def loading_embed(message: str = "リクエストを処理中…") -> discord.Embed:
    """処理中 Embed のショートカット。"""
    return make_embed("処理中", message, colour=Colour.INFO, emoji=EMOJI["loading"])
