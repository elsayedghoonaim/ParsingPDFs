import aiohttp
import asyncio
import random
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("pdf_extraction")

# HTTP status codes that are safe to retry
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class RateLimitError(RuntimeError):
    """Raised when the API returns a 429 rate-limit / quota-exceeded response."""
    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after  # seconds to wait before retrying


class BaseProvider(ABC):
    """Abstract base class for all VLM/OCR providers."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Send a single image with prompt. Return extracted text."""
        pass

    @abstractmethod
    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Send multiple images in one request. Return extracted text."""
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """Does this provider support multi-image requests?"""
        pass

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Estimate cost. Override in subclasses that know pricing. Return None if unknown."""
        return None

    async def _retry_request(self, request_func, max_retries: int = 3) -> str:
        """
        Execute an async request with exponential backoff retry.

        Handles two error categories:
          - RateLimitError (429): waits for the server-specified retry delay if provided,
            otherwise falls back to exponential backoff.
          - Other retryable errors (500/502/503/504): uses exponential backoff with jitter.

        Args:
            request_func: Async callable that performs the HTTP request.
            max_retries: Total number of attempts (including first).

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                return await request_func()

            except RateLimitError as e:
                last_exc = e
                if attempt >= max_retries - 1:
                    raise

                # Honour the server's suggested retry delay; fall back to exponential backoff
                if e.retry_after > 0:
                    sleep = e.retry_after
                else:
                    sleep = (2 ** attempt) + random.uniform(0.5, 1.5)

                logger.warning(
                    f"Rate limited (attempt {attempt + 1}/{max_retries}), "
                    f"waiting {sleep:.0f}s before retry..."
                )
                await asyncio.sleep(sleep)

            except Exception as e:
                last_exc = e
                error_str = str(e)

                # Detect retryable status codes embedded in the error message
                is_retryable = any(
                    f"error {code}" in error_str.lower() or f"{code}:" in error_str
                    for code in _RETRYABLE_STATUSES
                )

                if is_retryable and attempt < max_retries - 1:
                    sleep = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning(
                        f"Retryable error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {sleep:.1f}s"
                    )
                    await asyncio.sleep(sleep)
                else:
                    raise

        raise last_exc  # should not be reached, satisfies type checker
