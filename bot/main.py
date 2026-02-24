"""
Athena — Main Entry Point

Initialises the Discord bot, loads all cogs and services, and starts the event
loop.  Run with:  python -m bot.main
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands

from bot.config import settings, CACHE_DIR, LOGS_DIR
from bot.services.llm_service import LLMService
from bot.services.image_service import ImageService
from bot.services.queue_service import QueueService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
try:
    _handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
except OSError:
    pass  # Log dir not writable (e.g. Docker bind-mount as root) — stdout only

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=_handlers,
)
logger = logging.getLogger("athena")


# ---------------------------------------------------------------------------
# Bot class
# ---------------------------------------------------------------------------
class Athena(commands.Bot):
    """Main bot class with service lifecycle management."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix=settings.command_prefix,
            intents=intents,
            help_command=None,
        )

        # Services — initialised in setup_hook
        self.llm_service: Optional[LLMService] = None
        self.image_service: Optional[ImageService] = None
        self.queue_service: Optional[QueueService] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def setup_hook(self) -> None:
        """Called once before the bot connects.  Loads cogs & services."""
        logger.info("⚙  setup_hook: initialising services …")

        # Ensure cache directory exists
        try:
            Path(settings.image_cache_dir).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning("Cannot create image cache dir '%s' — using /tmp fallback", settings.image_cache_dir)
            settings.image_cache_dir = "/tmp/athena_cache/images"
            Path(settings.image_cache_dir).mkdir(parents=True, exist_ok=True)

        # Initialise services
        self.llm_service = LLMService()
        self.image_service = ImageService()
        self.queue_service = QueueService(
            llm_service=self.llm_service,
            image_service=self.image_service,
        )

        # Load cogs
        cog_modules = [
            "bot.cogs.chat",
            "bot.cogs.imagine",
            "bot.cogs.upscale",
            "bot.cogs.utility",
            "bot.cogs.admin",
        ]
        for module in cog_modules:
            try:
                await self.load_extension(module)
                logger.info("  ✓ Loaded cog: %s", module)
            except Exception:
                logger.exception("  ✗ Failed to load cog: %s", module)

        # Sync application commands globally
        logger.info("⚙  Syncing application commands …")
        await self.tree.sync()

    async def on_ready(self) -> None:
        """Fires when the bot has connected and is ready."""
        assert self.user is not None
        logger.info(
            "✦ Athena is online! %s (ID: %s)",
            self.user.name,
            self.user.id,
        )
        logger.info("  Guilds: %d", len(self.guilds))

        # Load models in background
        asyncio.create_task(self._load_models())

        # Set presence
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="/help • AI at your service",
            )
        )

    async def _load_models(self) -> None:
        """Load ML models in the background."""
        logger.info("🔄 Loading ML models in background …")

        loop = asyncio.get_running_loop()

        # Check / pull Ollama model (async)
        if self.llm_service:
            try:
                await self.llm_service.check_and_pull_model()
                logger.info("  ✓ LLM model ready (Ollama)")
            except Exception:
                logger.exception("  ✗ Failed to initialise LLM")

        # Load SD (blocking call — run in executor)
        if self.image_service:
            try:
                await loop.run_in_executor(None, self.image_service.load_model)
                logger.info("  ✓ Image generation model loaded")
            except Exception:
                logger.exception("  ✗ Failed to load Image model")

        # Start queue workers
        if self.queue_service:
            self.queue_service.start_workers()
            logger.info("  ✓ Queue workers started")

        logger.info("✅ All models loaded — Athena is fully operational")

    async def close(self) -> None:
        """Clean shutdown."""
        logger.info("🛑 Shutting down Athena …")

        # Stop queue workers
        if self.queue_service:
            await self.queue_service.stop_workers()

        # Close LLM HTTP session
        if self.llm_service:
            await self.llm_service.close()

        # Unload SD
        if self.image_service:
            self.image_service.unload_model()

        await super().close()
        logger.info("👋 Athena has shut down cleanly")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> None:
    """Synchronous entry-point."""
    bot = Athena()

    def _signal_handler(sig: int, _frame: object) -> None:
        logger.info("Received signal %s — scheduling shutdown", signal.Signals(sig).name)
        asyncio.ensure_future(bot.close())

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        bot.run(settings.discord_token, log_handler=None)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — exiting")


if __name__ == "__main__":
    main()
