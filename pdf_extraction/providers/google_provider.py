import base64
import json
import logging
import aiohttp
import os
from typing import Optional
from .base import BaseProvider, RateLimitError

logger = logging.getLogger("pdf_extraction")


def _parse_retry_delay(error_json: dict) -> float:
    """Extract the server-suggested retry delay (in seconds) from a Google API 429 response."""
    for detail in error_json.get("error", {}).get("details", []):
        if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
            delay_str = detail.get("retryDelay", "0s")
            try:
                return float(delay_str.rstrip("s"))
            except ValueError:
                pass
    return 0.0


class GoogleProvider(BaseProvider):
    """Google Gemini VLM provider supporting Gemini 2.0 Flash, 1.5 Pro, etc."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.provider_base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = (
            config.provider_api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self.model = config.provider_model

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY env var, "
                "or provider.api_key in config."
            )

    def _build_headers(self) -> dict:
        """API key sent via x-goog-api-key header (not URL param) to avoid leaking in logs."""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def _model_name(self) -> str:
        model = self.model
        if model.startswith("models/"):
            model = model[len("models/"):]
        return model

    def _build_url(self) -> str:
        return f"{self.base_url}/models/{self._model_name()}:generateContent"

    def _build_single_payload(self, image_bytes: bytes, prompt: str) -> dict:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64_image}}
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": self.config.vlm_max_output_tokens}
        }

    def _build_batch_payload(self, images: list[bytes], prompt: str) -> dict:
        parts = [{"text": prompt}]
        for image_bytes in images:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64_image}})
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {"maxOutputTokens": self.config.vlm_max_output_tokens}
        }

    @staticmethod
    def _parse_response(result: dict) -> str:
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Google API response format: {e}. Got: {result}")

    async def _handle_error_response(self, response: aiohttp.ClientResponse) -> None:
        """Parse an error response and raise the appropriate exception (clean log lines)."""
        status = response.status
        try:
            error_json = await response.json(content_type=None)
        except Exception:
            raw = await response.text()
            logger.debug(f"Google API error body (status {status}): {raw}")
            raise RuntimeError(f"Google API error {status} (see DEBUG log for details)")

        error_msg = error_json.get("error", {}).get("message", "")
        # Log a short human-readable summary at WARNING/ERROR level
        short_msg = error_msg.split("\n")[0][:200] if error_msg else f"HTTP {status}"
        logger.debug(f"Google API error body: {json.dumps(error_json, indent=2)}")

        if status == 429:
            retry_after = _parse_retry_delay(error_json)
            logger.warning(
                f"Google API rate limited — {short_msg} "
                f"(retry in {retry_after:.0f}s)"
            )
            raise RateLimitError(f"Google API rate limited", retry_after=retry_after)

        logger.error(f"Google API error {status}: {short_msg}")
        raise RuntimeError(f"Google API error {status}: {short_msg}")

    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a single image to Google's generateContent endpoint with retry."""
        url = self._build_url()
        payload = self._build_single_payload(image_bytes, prompt)
        headers = self._build_headers()

        async def _request():
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    await self._handle_error_response(response)
                result = await response.json()
                return self._parse_response(result)

        return await self._retry_request(_request)

    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a batch of images to Google's generateContent endpoint with retry."""
        url = self._build_url()
        payload = self._build_batch_payload(images, prompt)
        headers = self._build_headers()

        async def _request():
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    await self._handle_error_response(response)
                result = await response.json()
                return self._parse_response(result)

        return await self._retry_request(_request)

    def supports_batch(self) -> bool:
        return True

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        if self.config.cost_price_per_image is not None:
            return num_images * self.config.cost_price_per_image

        model_lower = self.model.lower()
        if "pro" in model_lower:
            price = 0.007
        elif "flash" in model_lower:
            price = 0.00015
        else:
            price = 0.00015
        return num_images * price
