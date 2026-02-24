"""
Athena — LLM Inference Service

Communicates with a local Ollama instance for text generation.  Ollama handles
GPU allocation, model management, and GGUF/safetensors loading natively.

All generation methods are async — they call the Ollama HTTP API directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import aiohttp
from pathlib import Path

from bot.config import settings, CACHE_DIR

CONVERSATIONS_DIR = CACHE_DIR / "conversations"

logger = logging.getLogger("athena.llm")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMModelNotLoaded(LLMError):
    pass


class LLMAPIError(LLMError):
    def __init__(self, status: int, message: str):
        self.message = message
        super().__init__(f"Ollama API error ({status}): {message}")


class LLMTimeoutError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


# ---------------------------------------------------------------------------
# Conversation history helpers
# ---------------------------------------------------------------------------
@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    timestamp: float = 0.0  # Optional: for sorting or display


@dataclass
class ConversationHistory:
    """Per-user conversation buffer."""

    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def trim(self, max_turns: int) -> None:
        """Keep at most *max_turns* user+assistant pairs (system always kept)."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]
        max_messages = max_turns * 2
        if len(other_msgs) > max_messages:
            other_msgs = other_msgs[-max_messages:]
        self.messages = system_msgs + other_msgs

    def to_chat_format(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_dict(self) -> dict:
        return {"messages": [{"role": m.role, "content": m.content} for m in self.messages]}

    @classmethod
    def from_dict(cls, data: dict) -> ConversationHistory:
        history = cls()
        if "messages" in data:
            history.messages = [Message(role=m["role"], content=m["content"]) for m in data["messages"]]
        return history

    def clear(self) -> None:
        self.messages.clear()


class _LRUConversationCache(OrderedDict[int, ConversationHistory]):
    """Bounded OrderedDict that evicts the oldest entry on overflow."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __getitem__(self, key: int) -> ConversationHistory:
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key: int, value: ConversationHistory) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            oldest_key = next(iter(self))
            del self[oldest_key]


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "あなたは Athena、Discord に組み込まれた親切で知識豊富な AI アシスタントです。"
    "質問には正確かつ簡潔に答えてください。わからない場合はそう伝えてください。"
    "ユーザーが使用する言語で会話してください。"
    "回答は Discord のメッセージ制限に合わせて 1900 文字以内に収めてください。"
)


class LLMService:
    """Manages LLM inference via a local Ollama instance."""

    def __init__(self) -> None:
        self._conversations = _LRUConversationCache(maxsize=settings.max_users_cached)
        self._loaded = False
        self._load_status = "未ロード"
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Ensure conversation directory exists
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self._session

    async def check_and_pull_model(self) -> None:
        """
        Check if the configured model is available in Ollama.
        If not, pull it.  This is called on bot startup.
        """
        session = await self.ensure_session()
        base_url = settings.ollama_base_url

        # Check if model exists
        try:
            async with session.post(
                f"{base_url}/api/show",
                json={"model": settings.ollama_model},
            ) as resp:
                if resp.status == 200:
                    logger.info("✓ Model '%s' is available in Ollama", settings.ollama_model)
                    self._loaded = True
                    return
        except aiohttp.ClientError:
            pass

        # Model not found — pull it
        logger.info("⬇ Pulling model '%s' from Ollama …", settings.ollama_model)
        self._load_status = "モデルをダウンロード中 (0%)..."
        
        try:
            async with session.post(
                f"{base_url}/api/pull",
                json={"model": settings.ollama_model, "stream": True},
            ) as resp:
                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "pulling" in status or "downloading" in status:
                                completed = data.get("completed", 0)
                                total = data.get("total", 0)
                                if total > 0:
                                    pct = completed / total * 100
                                    logger.info("  ⬇ %s: %.1f%%", status, pct)
                                    self._load_status = f"モデルをダウンロード中 ({pct:.0f}%)..."
                                else:
                                    logger.info("  ⬇ %s", status)
                                    self._load_status = f"モデルをダウンロード中 ({status})..."
                            elif status == "success":
                                logger.info("  ✓ Model pull complete")
                                self._load_status = "モデルの読み込み完了"
                        except json.JSONDecodeError:
                            pass

            self._loaded = True
            logger.info("✓ Model '%s' pulled successfully", settings.ollama_model)

        except Exception:
            logger.exception("✗ Failed to pull model '%s'", settings.ollama_model)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Conversation management ──────────────────────────────────────────

    def _get_history_path(self, user_id: int) -> Path:
        return CONVERSATIONS_DIR / f"{user_id}.json"

    def _save_history(self, user_id: int, history: ConversationHistory) -> None:
        """Save history to disk (synchronous but fast for small JSONs)."""
        try:
            path = self._get_history_path(user_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history.to_dict(), f, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to save history for user %s", user_id)

    def _load_history(self, user_id: int) -> Optional[ConversationHistory]:
        """Load history from disk."""
        path = self._get_history_path(user_id)
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ConversationHistory.from_dict(data)
        except Exception:
            logger.exception("Failed to load history for user %s", user_id)
            return None

    def get_history(self, user_id: int) -> ConversationHistory:
        if user_id not in self._conversations:
            # Try loading from disk
            history = self._load_history(user_id)
            if history:
                self._conversations[user_id] = history
            else:
                history = ConversationHistory()
                history.add("system", SYSTEM_PROMPT)
                self._conversations[user_id] = history
        return self._conversations[user_id]

    def clear_history(self, user_id: int) -> None:
        if user_id in self._conversations:
            del self._conversations[user_id]
        
        # Delete from disk
        path = self._get_history_path(user_id)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("Failed to delete history file for user %s", user_id)

    # ── Generation ───────────────────────────────────────────────────────

    async def generate(
        self,
        user_id: int,
        prompt: str,
        *,
        system_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response for *prompt* within the user's conversation
        context via Ollama's /api/chat endpoint.

        Returns the assistant's reply as a string.
        """

        if not self._loaded:
            raise LLMModelNotLoaded(f"LLMモデルを準備中です: {self._load_status}")

        session = await self.ensure_session()

        # Build messages
        if system_override:
            messages = [
                {"role": "system", "content": system_override},
                {"role": "user", "content": prompt},
            ]
        else:
            history = self.get_history(user_id)
            history.add("user", prompt)
            history.trim(settings.max_conversation_turns)
            messages = history.to_chat_format()

        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens if max_tokens is not None else settings.llm_max_tokens,
                "temperature": temperature if temperature is not None else settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "repeat_penalty": settings.llm_repeat_penalty,
            },
        }

        try:
            async with session.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Ollama API error (%d): %s", resp.status, error_text)
                    raise LLMAPIError(resp.status, error_text)

                data = await resp.json()
                content = data.get("message", {}).get("content", "")

        except asyncio.TimeoutError:
            logger.error("Ollama API timeout")
            raise LLMTimeoutError("Request timed out")
        except aiohttp.ClientError as exc:
            logger.error("Ollama connection error: %s", exc)
            raise LLMConnectionError(f"Connection failed: {exc}")

        # Save to history
        if not system_override:
            history.add("assistant", content)
            self._save_history(user_id, history)

        return content.strip()

    async def generate_stream(
        self,
        user_id: int,
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant — yields text chunks.
        """

        if not self._loaded:
            raise LLMModelNotLoaded(f"LLMモデルを準備中です: {self._load_status}")

        session = await self.ensure_session()
        history = self.get_history(user_id)
        history.add("user", prompt)
        history.trim(settings.max_conversation_turns)

        payload = {
            "model": settings.ollama_model,
            "messages": history.to_chat_format(),
            "stream": True,
            "options": {
                "num_predict": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "repeat_penalty": settings.llm_repeat_penalty,
            },
        }

        full_response: list[str] = []

        try:
            async with session.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Ollama API error (%d): %s", resp.status, error_text)
                    raise LLMAPIError(resp.status, error_text)

                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                full_response.append(token)
                                yield token
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.exception("Streaming generation failed")
            raise LLMError(f"Streaming failed: {e}")

        history.add("assistant", "".join(full_response))
        self._save_history(user_id, history)
