import base64
import logging
import aiohttp
import os
from typing import Optional
from .base import BaseProvider

logger = logging.getLogger("pdf_extraction")


class GoogleProvider(BaseProvider):
    """Google Gemini VLM provider supporting Gemini 1.5 Flash, Pro, etc."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.provider_base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = config.provider_api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = config.provider_model

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY env var, or provider.api_key in config."
            )

    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a single image to Google's generateContent endpoint."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": 4096}
        }

        # Google Gemini model name needs to omit the models/ prefix if it exists in base_url or path
        model_name = self.model
        if model_name.startswith("models/"):
            model_name = model_name.replace("models/", "")

        url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"

        try:
            async with session.post(url, json=payload) as response:
                status = response.status
                response_text = await response.text()
                if status != 200:
                    logger.error(f"Google API error {status}: {response_text}")
                    raise RuntimeError(f"Google API error {status}: {response_text}")
                
                result = await response.json()
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as e:
                    logger.error(f"Invalid Google API response format: {result}. Error: {e}")
                    raise RuntimeError(f"Invalid Google API response format: {e}")
        except Exception as e:
            logger.error(f"Failed to send image to Google Gemini: {e}")
            raise

    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a batch of images to Google's generateContent endpoint in a single request."""
        parts = [{"text": prompt}]
        for image_bytes in images:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": b64_image
                }
            })

        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {"maxOutputTokens": 4096}
        }

        model_name = self.model
        if model_name.startswith("models/"):
            model_name = model_name.replace("models/", "")

        url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"

        try:
            async with session.post(url, json=payload) as response:
                status = response.status
                response_text = await response.text()
                if status != 200:
                    logger.error(f"Google API batch error {status}: {response_text}")
                    raise RuntimeError(f"Google API batch error {status}: {response_text}")
                
                result = await response.json()
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as e:
                    logger.error(f"Invalid Google API response format: {result}. Error: {e}")
                    raise RuntimeError(f"Invalid Google API response format: {e}")
        except Exception as e:
            logger.error(f"Failed to send image batch to Google Gemini: {e}")
            raise

    def supports_batch(self) -> bool:
        """Google Gemini models naturally support batching multiple images in a single call."""
        return True

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Estimates cost based on model and number of images/pages."""
        if self.config.cost_price_per_image is not None:
            return num_images * self.config.cost_price_per_image

        model_lower = self.model.lower()
        if "pro" in model_lower:
            price = 0.007  # ~$0.007 per image
        elif "flash" in model_lower:
            price = 0.00015  # ~$0.00015 per image
        else:
            price = 0.00015  # Default to Flash pricing
        return num_images * price
