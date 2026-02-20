"""
Athena — Queue / Task Management Service

Provides bounded async queues for LLM and image generation requests.
LLM requests are natively async (Ollama HTTP API).
Image requests are blocking (diffusers) and run in an executor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from bot.config import settings
from bot.services.llm_service import LLMService
from bot.services.image_service import ImageService

logger = logging.getLogger("athena.queue")


class TaskType(Enum):
    LLM = "llm"
    IMAGE = "image"


@dataclass
class TaskRequest:
    """A unit of work to be processed."""

    task_type: TaskType
    # For IMAGE: a sync callable.  For LLM: an async coroutine function.
    func: Any
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    future: asyncio.Future[Any] = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    submitted_at: float = field(default_factory=time.time)

    @property
    def wait_seconds(self) -> float:
        return time.time() - self.submitted_at


class QueueService:
    """Async task queue manager for LLM and image generation workloads."""

    def __init__(
        self,
        llm_service: LLMService,
        image_service: ImageService,
    ) -> None:
        self._llm_service = llm_service
        self._image_service = image_service

        self._llm_queue: asyncio.Queue[TaskRequest] = asyncio.Queue(
            maxsize=settings.llm_queue_max
        )
        self._image_queue: asyncio.Queue[TaskRequest] = asyncio.Queue(
            maxsize=settings.image_queue_max
        )

        self._workers: list[asyncio.Task[None]] = []

        # Stats
        self.llm_processed: int = 0
        self.image_processed: int = 0
        self.llm_total_time: float = 0.0
        self.image_total_time: float = 0.0

    # ── Queue submission ─────────────────────────────────────────────────

    async def submit_llm(
        self,
        coro_func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Submit an async LLM coroutine function to the queue.
        Raises ``asyncio.QueueFull`` if the queue is at capacity.
        """
        loop = asyncio.get_running_loop()
        request = TaskRequest(
            task_type=TaskType.LLM,
            func=coro_func,
            args=args,
            kwargs=kwargs,
            future=loop.create_future(),
        )

        try:
            self._llm_queue.put_nowait(request)
        except asyncio.QueueFull:
            raise asyncio.QueueFull(
                f"LLM queue is full ({settings.llm_queue_max} pending). "
                "Please try again later."
            )

        return await request.future

    async def submit_image(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Submit a blocking image generation function to the queue."""
        loop = asyncio.get_running_loop()
        request = TaskRequest(
            task_type=TaskType.IMAGE,
            func=func,
            args=args,
            kwargs=kwargs,
            future=loop.create_future(),
        )

        try:
            self._image_queue.put_nowait(request)
        except asyncio.QueueFull:
            raise asyncio.QueueFull(
                f"Image queue is full ({settings.image_queue_max} pending). "
                "Please try again later."
            )

        return await request.future

    # ── Workers ──────────────────────────────────────────────────────────

    def start_workers(self) -> None:
        """Start background consumer tasks for each queue."""
        loop = asyncio.get_event_loop()
        self._workers = [
            loop.create_task(self._llm_worker()),
            loop.create_task(self._image_worker()),
        ]
        logger.info("Queue workers started (LLM + Image)")

    async def stop_workers(self) -> None:
        """Cancel worker tasks."""
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Queue workers stopped")

    async def _llm_worker(self) -> None:
        """LLM consumer — calls async coroutines directly (Ollama is async)."""
        logger.info("LLM worker ready")

        while True:
            try:
                request = await self._llm_queue.get()
            except asyncio.CancelledError:
                break

            t0 = time.perf_counter()
            try:
                # func is an async coroutine function
                result = await request.func(*request.args, **request.kwargs)
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as exc:
                if not request.future.done():
                    request.future.set_exception(exc)
                logger.exception("LLM worker: task failed")
            finally:
                elapsed = time.perf_counter() - t0
                self.llm_processed += 1
                self.llm_total_time += elapsed
                self._llm_queue.task_done()
                logger.debug("LLM worker: completed in %.1fs", elapsed)

    async def _image_worker(self) -> None:
        """Image consumer — runs blocking functions in executor."""
        logger.info("Image worker ready")
        loop = asyncio.get_running_loop()

        while True:
            try:
                request = await self._image_queue.get()
            except asyncio.CancelledError:
                break

            t0 = time.perf_counter()
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda r=request: r.func(*r.args, **r.kwargs),
                )
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as exc:
                if not request.future.done():
                    request.future.set_exception(exc)
                logger.exception("Image worker: task failed")
            finally:
                elapsed = time.perf_counter() - t0
                self.image_processed += 1
                self.image_total_time += elapsed
                self._image_queue.task_done()
                logger.debug("Image worker: completed in %.1fs", elapsed)

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def llm_queue_size(self) -> int:
        return self._llm_queue.qsize()

    @property
    def image_queue_size(self) -> int:
        return self._image_queue.qsize()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "llm_queue_size": self.llm_queue_size,
            "llm_queue_max": settings.llm_queue_max,
            "llm_processed": self.llm_processed,
            "llm_avg_time": (
                self.llm_total_time / self.llm_processed
                if self.llm_processed
                else 0.0
            ),
            "image_queue_size": self.image_queue_size,
            "image_queue_max": settings.image_queue_max,
            "image_processed": self.image_processed,
            "image_avg_time": (
                self.image_total_time / self.image_processed
                if self.image_processed
                else 0.0
            ),
        }
