import base64
import logging
import aiohttp
import os
from typing import Optional
from .base import BaseProvider

logger = logging.getLogger("pdf_extraction")


class CustomProvider(BaseProvider):
    """Custom VLM provider with user-defined base URL, headers, and batch capabilities."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.custom_base_url
        self.api_key = config.custom_api_key or os.environ.get("CUSTOM_API_KEY") or config.provider_api_key or os.environ.get("PROVIDER_API_KEY") or ""
        self.model = config.custom_model
        self.extra_headers = config.custom_headers or {}
        self._supports_batch = config.custom_supports_batch

    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a single image to the custom endpoint using OpenAI-compatible payload format."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Merge custom user headers
        headers.update(self.extra_headers)

        url = f"{self.base_url}/chat/completions"

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                status = response.status
                response_text = await response.text()
                if status != 200:
                    logger.error(f"Custom VLM API error {status}: {response_text}")
                    raise RuntimeError(f"Custom VLM API error {status}: {response_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to send image to custom provider: {e}")
            raise

    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends batch of images to custom endpoint if batching is supported, else sequential fallback."""
        if not self._supports_batch:
            logger.warning("Custom provider is configured without batch support. Falling back to sequential calls.")
            results = []
            for i, img_bytes in enumerate(images):
                try:
                    res = await self.send_image(img_bytes, prompt, session)
                    results.append(res)
                except Exception as e:
                    logger.error(f"Error in Custom VLM batch sequential call at page index {i}: {e}")
                    results.append(f"[Custom VLM Error: Page extraction failed: {e}]")
            return "\n\n---\n\n".join(results)

        content_payload = [{"type": "text", "text": prompt}]
        for image_bytes in images:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"}
            })

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content_payload
                }
            ],
            "max_tokens": 4096
        }

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Merge custom user headers
        headers.update(self.extra_headers)

        url = f"{self.base_url}/chat/completions"

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                status = response.status
                response_text = await response.text()
                if status != 200:
                    logger.error(f"Custom VLM API batch error {status}: {response_text}")
                    raise RuntimeError(f"Custom VLM API batch error {status}: {response_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to send image batch to custom provider: {e}")
            raise

    def supports_batch(self) -> bool:
        """Configured by custom_supports_batch setting."""
        return self._supports_batch

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Custom endpoint estimation using the cost_price_per_image override if provided."""
        if self.config.cost_price_per_image is not None:
            return num_images * self.config.cost_price_per_image
        return None
