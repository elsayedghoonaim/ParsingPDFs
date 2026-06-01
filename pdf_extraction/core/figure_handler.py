"""Figure description runner — describes figures extracted in PageResult objects using a VLM provider."""

import asyncio
import aiohttp
import logging
import os
from . import PageResult, PipelineConfig
from .postprocessor import clean_vlm_output
from ..providers.base import BaseProvider

logger = logging.getLogger("pdf_extraction")


def _load_figure_prompt() -> str:
    """
    Load the figure description prompt template from prompts/figure_description.txt.
    """
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts"
    )
    prompt_path = os.path.join(prompts_dir, "figure_description.txt")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load figure description prompt: {e}")
        return "Describe this figure in detail, focusing on charts, data trends, labels, and visual content."


async def vlm_describe_figures(
    page_results: list[PageResult],
    provider: BaseProvider,
    config: PipelineConfig
) -> tuple[list[PageResult], int]:
    """
    Concurrently describe all extracted figures across all pages.
    
    Returns the updated PageResult list and the total number of VLM calls made.
    """
    prompt = _load_figure_prompt()
    sem = asyncio.Semaphore(config.max_concurrent_requests)
    
    # Collect all figures across all pages that need descriptions
    figure_tasks = []  # list of (page_idx, figure_idx, image_bytes)
    for page_idx, pr in enumerate(page_results):
        for fig_idx, fig in enumerate(pr.figures_metadata):
            image_bytes = fig.get("image_bytes")
            if image_bytes and len(image_bytes) > 0:
                figure_tasks.append((page_idx, fig_idx, image_bytes))
                
    if not figure_tasks:
        logger.info("No figures needing VLM descriptions found in page results.")
        return page_results, 0

    vlm_calls = 0
    vlm_calls_lock = asyncio.Lock()

    async def describe_one(page_idx: int, fig_idx: int, image_bytes: bytes) -> tuple[int, int, str]:
        nonlocal vlm_calls
        async with sem:
            client_timeout = aiohttp.ClientTimeout(total=config.vlm_timeout_seconds)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                try:
                    description = await provider.send_image(image_bytes, prompt, session)
                    async with vlm_calls_lock:
                        vlm_calls += 1
                    if config.postprocessing_enabled:
                        description = clean_vlm_output(description, config)
                    return page_idx, fig_idx, description
                except Exception as e:
                    logger.warning(f"Failed to describe figure {fig_idx} on page {page_idx}: {e}")
                    return page_idx, fig_idx, "[Figure description unavailable]"

    tasks = [describe_one(pi, fi, ib) for pi, fi, ib in figure_tasks]
    results = await asyncio.gather(*tasks)
    
    # Apply descriptions back to the page results
    for page_idx, fig_idx, description in results:
        page_results[page_idx].figures_metadata[fig_idx]["description"] = description
        page_results[page_idx].figure_descriptions.append(description)
        
    logger.info(f"Described {len(results)} figures across {len(page_results)} pages")
    return page_results, vlm_calls
