"""VLM/OCR Extraction module — performs multimodal text extraction using VLMs."""

import fitz
import asyncio
import aiohttp
import logging
import os
import re
from . import PageResult, PipelineConfig
from ..providers.base import BaseProvider
from .postprocessor import clean_vlm_output

logger = logging.getLogger("pdf_extraction")


def _load_prompt(prompt_filename: str, optional_prompt: str | None = None) -> str:
    """
    Load a prompt template from the prompts directory and append optional custom instructions.
    """
    # prompts directory is in the parent of the core directory
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts"
    )
    prompt_path = os.path.join(prompts_dir, prompt_filename)
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load prompt file {prompt_filename}: {e}")
        # Return a sensible fallback prompt
        prompt = "You are a precise document OCR system. Transcribe all text in this image into markdown format."
        
    if optional_prompt and optional_prompt.strip():
        prompt = prompt + "\n\n" + optional_prompt.strip()
        
    return prompt


def _render_page_to_image(page: fitz.Page, dpi: int) -> bytes:
    """
    Render a single PDF page into a PNG image at the given DPI.
    """
    # 72 is the default PDF point system DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


async def extract_scanned_markdown(
    pdf_bytes: bytes,
    provider: BaseProvider,
    config: PipelineConfig
) -> tuple[list[PageResult], int]:
    """
    Run multimodal OCR on all pages of a scanned PDF.
    
    Renders pages to images lazily, routes them using either single-page or batch-page strategy,
    sends them concurrently to the VLM provider under a concurrency semaphore,
    handles errors gracefully on a per-page/per-batch basis, and post-processes the output.
    
    Returns:
        tuple[list[PageResult], int]: (page results, total VLM calls made)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = len(doc)
        if page_count == 0:
            logger.warning("Empty PDF document provided to VLM extractor.")
            return [], 0

        # 1. Determine prompt file based on configuration
        if config.vlm_describe_figures:
            prompt_filename = "ocr_with_figures.txt"
        else:
            prompt_filename = "ocr_text_only.txt"
            
        prompt = _load_prompt(prompt_filename, config.optional_prompt)

        # If batching, inject a rule about separating pages using [===PAGE_BREAK===]
        if config.vlm_page_batching == "batch":
            batch_delimiter_instruction = (
                "\n\nIMPORTANT: Since you are processing multiple page images, "
                "you MUST separate the Markdown content of each page using exactly the delimiter: "
                "\n[===PAGE_BREAK===]\n"
                "Do not add any other separator between pages, and output the delimiter on its own line."
            )
            prompt += batch_delimiter_instruction

        # 2. Setup concurrency controls
        sem = asyncio.Semaphore(config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=config.vlm_timeout_seconds)
        vlm_calls = 0
        vlm_calls_lock = asyncio.Lock()

        # Define single page processing helper with lazy rendering inside semaphore
        async def process_single_page(page_num: int) -> PageResult:
            nonlocal vlm_calls
            async with sem:
                try:
                    image_bytes = _render_page_to_image(doc[page_num], config.vlm_render_dpi)
                except Exception as e:
                    logger.error(f"Failed to render page {page_num} to image: {e}")
                    return PageResult(
                        page_num=page_num,
                        markdown_text="[OCR FAILED: Page rendering failed]"
                    )

                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        text = await provider.send_image(image_bytes, prompt, session)
                        async with vlm_calls_lock:
                            vlm_calls += 1
                        return PageResult(page_num=page_num, markdown_text=text)
                except Exception as e:
                    logger.error(f"Failed to extract page {page_num} via VLM: {e}", exc_info=True)
                    async with vlm_calls_lock:
                        # Count the attempted call
                        vlm_calls += 1
                    return PageResult(
                        page_num=page_num,
                        markdown_text=f"[OCR FAILED: {str(e)}]"
                    )

        page_results = []

        # 3. Route based on batching strategy
        if config.vlm_page_batching == "batch":
            if not provider.supports_batch():
                logger.warning("Provider does not support batch mode. Falling back to single mode.")
                tasks = [process_single_page(i) for i in range(page_count)]
                page_results = await asyncio.gather(*tasks)
            else:
                # Divide pages into batches
                batches = []
                for i in range(0, page_count, config.max_pages_per_batch):
                    batch_indices = list(range(i, min(i + config.max_pages_per_batch, page_count)))
                    batches.append((i, batch_indices))

                async def process_batch(batch_start: int, batch_indices: list[int]) -> list[PageResult]:
                    nonlocal vlm_calls
                    batch_results = []
                    
                    async with sem:
                        # Render pages in this batch
                        valid_imgs = []
                        failed_rendering_indices = set()
                        for page_num in batch_indices:
                            try:
                                page_img = _render_page_to_image(doc[page_num], config.vlm_render_dpi)
                                valid_imgs.append(page_img)
                            except Exception as e:
                                logger.error(f"Failed to render page {page_num} to image: {e}")
                                failed_rendering_indices.add(page_num)
                                valid_imgs.append(b"")

                        # If all pages failed rendering
                        if not any(valid_imgs):
                            for page_num in batch_indices:
                                batch_results.append(PageResult(
                                    page_num=page_num,
                                    markdown_text="[OCR FAILED: Page rendering failed]"
                                ))
                            return batch_results

                        # Map non-failed images and page numbers
                        non_failed_indices_and_imgs = [(p_num, img) for p_num, img in zip(batch_indices, valid_imgs) if img]
                        non_failed_p_nums = [item[0] for item in non_failed_indices_and_imgs]
                        non_failed_imgs = [item[1] for item in non_failed_indices_and_imgs]

                        # Add failed ones directly to results
                        for p_num in batch_indices:
                            if p_num in failed_rendering_indices:
                                batch_results.append(PageResult(
                                    page_num=p_num,
                                    markdown_text="[OCR FAILED: Page rendering failed]"
                                ))

                        try:
                            async with aiohttp.ClientSession(timeout=timeout) as session:
                                batch_text = await provider.send_images_batch(non_failed_imgs, prompt, session)
                                async with vlm_calls_lock:
                                    vlm_calls += 1
                                
                                # Split the combined text block using custom delimiter first, fall back to regex
                                if "[===PAGE_BREAK===]" in batch_text:
                                    chunks = re.split(r'\s*\[===PAGE_BREAK===\]\s*', batch_text)
                                else:
                                    chunks = re.split(r'\n(?:---|___|\*\*\*)\n|\f', batch_text)
                                chunks = [c.strip() for c in chunks]
                                
                                # Align chunks with non-failed pages in this batch
                                for j, page_num in enumerate(non_failed_p_nums):
                                    if j < len(chunks):
                                        chunk_text = chunks[j]
                                    else:
                                        chunk_text = "[OCR FAILED: Batch output parsing error - missing page chunk]"
                                    
                                    # If the VLM produced extra trailing chunks, append them to the last page
                                    if j == len(non_failed_p_nums) - 1 and len(chunks) > len(non_failed_p_nums):
                                        chunk_text += "\n\n" + "\n\n---\n\n".join(chunks[j+1:])
                                        
                                    batch_results.append(PageResult(
                                        page_num=page_num,
                                        markdown_text=chunk_text
                                    ))
                        except Exception as e:
                            logger.error(f"Failed to process batch starting at {batch_start}: {e}", exc_info=True)
                            async with vlm_calls_lock:
                                vlm_calls += 1
                            # All non-failed pages in this batch marked as failed
                            for page_num in non_failed_p_nums:
                                batch_results.append(PageResult(
                                    page_num=page_num,
                                    markdown_text=f"[OCR FAILED: Batch processing failed - {str(e)}]"
                                ))
                    return batch_results

                batch_tasks = [process_batch(start, indices) for start, indices in batches]
                all_batch_results = await asyncio.gather(*batch_tasks)
                for br in all_batch_results:
                    page_results.extend(br)
        else:
            # Single page processing mode
            tasks = [process_single_page(i) for i in range(page_count)]
            page_results = await asyncio.gather(*tasks)

        # 4. Sort page results by page number to guarantee correct ordering
        page_results = sorted(page_results, key=lambda r: r.page_num)

        # 5. Apply post-processing formatting if enabled
        if config.postprocessing_enabled:
            for pr in page_results:
                # Keep failed OCR placeholders as is, postprocess the others
                if not pr.markdown_text.startswith("[OCR FAILED:"):
                    pr.markdown_text = clean_vlm_output(pr.markdown_text, config)

        logger.info(f"Extracted {len(page_results)} pages via VLM ({vlm_calls} API calls)")
        return page_results, vlm_calls

    finally:
        doc.close()
