"""Public API interface for the PDF-to-Markdown extraction pipeline."""

import asyncio
import os
import sys
import time
import glob
import logging
import re
from typing import Optional

from .config import load_config, setup_logging
from .core import PageResult, ClassificationResult, ProcessResult, DryRunResult, PipelineConfig
from .core.classifier import classify_document
from .core.extractor_text import extract_text_markdown
from .core.extractor_vlm import extract_scanned_markdown
from .core.figure_handler import vlm_describe_figures
from .core.merger import merge_to_document
from .core.state import PipelineState
from .providers import get_provider


def _sanitize_filename(filename: str) -> str:
    """Sanitize the filename to prevent path traversal and reserved name conflicts on Windows."""
    # 1. Remove characters that are illegal in Windows/Unix filenames: < > : " / \ | ? *
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 2. Strip leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # 3. Check for Windows reserved device names (case-insensitive)
    name_without_ext, ext = os.path.splitext(filename)
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    if name_without_ext.upper() in reserved_names:
        filename = f"safe_{filename}"
        
    if not filename:
        filename = "safe_document"
        
    return filename


class PDFPipeline:
    """
    The main public class for the PDF Extraction Pipeline library.
    
    Manages lazy loading of providers, checks Windows OS event loop policies,
    and executes processes synchronously wrapping async calls.
    """

    def __init__(self, config_path: str = None, **config_overrides):
        """
        Initialize the PDFPipeline.

        Args:
            config_path (str, optional): Path to a user config.yaml.
                If None, the pipeline automatically looks for 'config.yaml' in
                the current working directory before falling back to built-in defaults.
            **config_overrides: Flat configuration keys to override defaults.
        """
        # Auto-detect config.yaml in the current working directory if not specified
        if config_path is None:
            cwd_config = os.path.join(os.getcwd(), "config.yaml")
            if os.path.exists(cwd_config):
                config_path = cwd_config
                print(f"[INFO] Auto-detected config: {cwd_config}")

        self.config = load_config(config_path, **config_overrides)
        self.logger = setup_logging(self.config)
        self.state = PipelineState(self.config.output_state_directory)
        
        # Only create provider when needed (lazy loaded)
        self._provider = None

    def _config_error(self, err: Exception, doc_id: str, source_path: str, start_time: float) -> "ProcessResult":
        """Return a clean failed ProcessResult and log a user-friendly config error."""
        self.logger.error(
            f"Provider not configured — {err}\n"
            f"  → Open config.yaml and set:\n"
            f"      provider.name:    google\n"
            f"      provider.api_key: <your-key>\n"
            f"      provider.model:   gemini-2.0-flash-lite\n"
            f"  → Or export GOOGLE_API_KEY=<your-key> in your terminal."
        )
        self.state.mark_failed(doc_id, str(err))
        return ProcessResult(
            doc_id=doc_id,
            source_path=source_path,
            output_path="",
            doc_type="",
            classification_strategy="",
            processing_time=time.time() - start_time,
            status="failed",
            error=f"Configuration error: {err}"
        )

    def _get_provider(self):
        """Lazy-load the provider — only when the first VLM call is needed."""
        if self._provider is None:
            self._provider = get_provider(self.config)
        return self._provider

    def _get_event_loop(self):
        """Get or create an event loop, safely handling Windows event loop policies."""
        if sys.platform == 'win32':
            try:
                # Set Windows Selector Event Loop Policy for aiohttp / asyncio subprocesses
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception as e:
                self.logger.debug(f"Could not set Windows event loop policy: {e}")
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            return None

    def process(self, pdf_path: str) -> ProcessResult:
        """
        Synchronously process a single PDF file on disk.
        
        Wraps the async processing pipeline safely.
        """
        loop = self._get_event_loop()
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._process_async(pdf_path))
                return future.result()
        else:
            return asyncio.run(self._process_async(pdf_path))

    def process_bytes(self, pdf_bytes: bytes, doc_id: str) -> ProcessResult:
        """
        Synchronously process PDF content directly from memory.
        
        Useful for pre-downloaded or streamed documents.
        """
        loop = self._get_event_loop()
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._process_bytes_async(pdf_bytes, doc_id))
                return future.result()
        else:
            return asyncio.run(self._process_bytes_async(pdf_bytes, doc_id))

    def process_directory(self, dir_path: str, resume: bool = True) -> list[ProcessResult]:
        """
        Synchronously process all PDF documents in a directory.
        
        Optionally skips already completed files if resume is True.
        """
        loop = self._get_event_loop()
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._process_directory_async(dir_path, resume))
                return future.result()
        else:
            return asyncio.run(self._process_directory_async(dir_path, resume))

    def dry_run(self, pdf_path: str) -> DryRunResult:
        """
        Analyze a PDF file without making any external VLM or API calls.
        
        Estimates total pages, visual elements, VLM calls, and API costs.
        """
        self.logger.info(f"Performing dry run on {pdf_path}")
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception as e:
            self.logger.error(f"Dry run failed to read file {pdf_path}: {e}")
            return DryRunResult(
                doc_id=PipelineState.generate_doc_id(pdf_path),
                doc_type="scanned"
            )

        doc_id = PipelineState.generate_doc_id(pdf_path)
        
        # Heuristic classification
        classification = classify_document(pdf_bytes, self.config)
        
        # Count pages and detect major figures using fitz
        import fitz
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            
            figures_detected = 0
            for page in doc:
                page_area = page.rect.width * page.rect.height
                if page_area > 0:
                    for img_info in page.get_image_info():
                        x0, y0, x1, y1 = img_info["bbox"]
                        # Filter for images covering at least 5% of the page
                        if ((x1 - x0) * (y1 - y0)) / page_area > 0.05:
                            figures_detected += 1
            doc.close()
        except Exception as e:
            self.logger.error(f"Dry run failed parsing PDF structures: {e}")
            total_pages = 0
            figures_detected = 0

        # Estimate VLM needs
        if classification.doc_type == "scanned":
            pages_needing_vlm = total_pages
            if self.config.vlm_page_batching == "batch":
                import math
                estimated_api_calls = math.ceil(total_pages / self.config.vlm_max_pages_per_batch)
            else:
                estimated_api_calls = total_pages
        elif self.config.vlm_describe_figures and figures_detected > 0:
            pages_needing_vlm = 0  # Text extraction is free, only VLM for figure descriptions
            estimated_api_calls = figures_detected
        else:
            pages_needing_vlm = 0
            estimated_api_calls = 0

        # Cost estimation
        estimated_cost = None
        if estimated_api_calls > 0 and self.config.cost_enabled:
            try:
                provider = self._get_provider()
                estimated_cost = provider.estimate_cost(estimated_api_calls, self.config.vlm_render_dpi)
            except Exception as e:
                self.logger.debug(f"Could not estimate cost during dry run: {e}")
                pass

        return DryRunResult(
            doc_id=doc_id,
            doc_type=classification.doc_type,
            total_pages=total_pages,
            pages_needing_vlm=pages_needing_vlm,
            estimated_api_calls=estimated_api_calls,
            estimated_cost=estimated_cost,
            figures_detected=figures_detected
        )

    async def _process_async(self, pdf_path: str) -> ProcessResult:
        """Core async flow for processing a single disk file."""
        start_time = time.time()
        doc_id = PipelineState.generate_doc_id(pdf_path)
        
        # Check if already completed and we are allowed to skip
        if self.state.is_completed(doc_id):
            self.logger.info(f"Skipping {pdf_path} — already completed successfully")
            return ProcessResult(
                doc_id=doc_id,
                source_path=pdf_path,
                output_path="",
                doc_type="",
                classification_strategy="",
                status="skipped"
            )

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            return await self._process_bytes_async(pdf_bytes, doc_id, source_path=pdf_path)
        except Exception as e:
            self.logger.error(f"Could not read file {pdf_path}: {e}")
            self.state.mark_failed(doc_id, str(e))
            return ProcessResult(
                doc_id=doc_id,
                source_path=pdf_path,
                output_path="",
                doc_type="",
                classification_strategy="",
                processing_time=time.time() - start_time,
                status="failed",
                error=str(e)
            )

    async def _process_bytes_async(self, pdf_bytes: bytes, doc_id: str, source_path: str = "") -> ProcessResult:
        """Core async flow for processing PDF bytes."""
        start_time = time.time()
        try:
            # 1. Classify the document type
            classification = classify_document(pdf_bytes, self.config)
            
            # 2. Extract content page by page based on classification type
            vlm_calls = 0
            
            if classification.doc_type == "extractable":
                # Path A: Extract text/markdown via PyMuPDF4LLM
                page_results = extract_text_markdown(pdf_bytes)
                
                # Optionally process embedded figure images using VLM
                if self.config.vlm_describe_figures and any(pr.figures_metadata for pr in page_results):
                    try:
                        provider = self._get_provider()
                    except ValueError as cfg_err:
                        return self._config_error(cfg_err, doc_id, source_path, start_time)
                    page_results, fig_calls = await vlm_describe_figures(page_results, provider, self.config)
                    vlm_calls += fig_calls

                # Apply post-processing cleanup to native text pages if enabled
                if self.config.postprocessing_enabled:
                    from .core.postprocessor import clean_vlm_output
                    for pr in page_results:
                        pr.markdown_text = clean_vlm_output(pr.markdown_text, self.config)
            else:
                # Path B: Full OCR/VLM extraction for scanned documents
                try:
                    provider = self._get_provider()
                except ValueError as cfg_err:
                    return self._config_error(cfg_err, doc_id, source_path, start_time)
                page_results, vlm_calls = await extract_scanned_markdown(pdf_bytes, provider, self.config)

            # 3. Merge pages into a single consolidated Markdown string
            final_markdown = merge_to_document(page_results, classification.doc_type)

            # 4. Save output markdown to disk
            os.makedirs(self.config.output_directory, exist_ok=True)
            if self.config.output_filename_strategy == "original" and source_path:
                base_name = os.path.splitext(os.path.basename(source_path))[0]
                sanitized_base = _sanitize_filename(base_name)
                output_filename = f"{sanitized_base}.md"
            else:
                output_filename = f"{doc_id}.md"

            output_path = os.path.join(self.config.output_directory, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_markdown)

            # 5. Estimate VLM costs if calls were made
            estimated_cost = None
            if vlm_calls > 0 and self.config.cost_enabled:
                try:
                    provider = self._get_provider()
                    estimated_cost = provider.estimate_cost(vlm_calls, self.config.vlm_render_dpi)
                    if estimated_cost and estimated_cost > self.config.cost_warn_threshold_usd:
                        self.logger.warning(
                            f"Estimated cost (${estimated_cost:.4f}) exceeds warning threshold "
                            f"(${self.config.cost_warn_threshold_usd:.2f})!"
                        )
                except Exception as e:
                    self.logger.debug(f"Could not calculate final execution cost: {e}")
                    pass

            processing_time = time.time() - start_time
            
            result = ProcessResult(
                doc_id=doc_id,
                source_path=source_path,
                output_path=output_path,
                doc_type=classification.doc_type,
                classification_strategy=self.config.classification_strategy,
                confidence_metrics=classification.confidence_metrics,
                pages_processed=len(page_results),
                vlm_calls_made=vlm_calls,
                estimated_cost=estimated_cost,
                processing_time=processing_time,
                status="completed"
            )

            # 6. Save completion state
            self.state.mark_completed(doc_id, {
                "source_path": source_path,
                "doc_type": classification.doc_type,
                "pages_processed": len(page_results),
                "vlm_calls_made": vlm_calls,
                "output_file": output_path
            })

            doc_name = os.path.basename(source_path) if source_path else doc_id[:8]
            self.logger.info(f"✓ {doc_name} → {output_path} ({processing_time:.1f}s, {len(page_results)}p, {vlm_calls} API calls)")
            return result

        except Exception as e:
            self.logger.error(f"Unexpected error processing {source_path or doc_id[:8]}: {e}")
            self.state.mark_failed(doc_id, str(e))
            return ProcessResult(
                doc_id=doc_id,
                source_path=source_path,
                output_path="",
                doc_type="",
                classification_strategy="",
                processing_time=time.time() - start_time,
                status="failed",
                error=str(e)
            )

    async def _process_directory_async(self, dir_path: str, resume: bool) -> list[ProcessResult]:
        """Core async flow for processing an entire directory of PDFs concurrently."""
        # Find all PDF files (case-insensitive globbing)
        pdf_files = glob.glob(os.path.join(dir_path, "*.pdf"))
        pdf_files += glob.glob(os.path.join(dir_path, "*.PDF"))
        pdf_files = list(sorted(set(pdf_files)))

        if not pdf_files:
            self.logger.warning(f"No PDF documents found in directory: {dir_path}")
            return []

        self.logger.info(f"Found {len(pdf_files)} PDFs in directory {dir_path}")

        # Filter out already processed documents if in resume mode
        if resume:
            pending_files = []
            skipped_count = 0
            for f in pdf_files:
                did = PipelineState.generate_doc_id(f)
                if self.state.is_completed(did):
                    skipped_count += 1
                else:
                    pending_files.append(f)
            
            if skipped_count > 0:
                self.logger.info(f"Resume Mode enabled: skipped {skipped_count} completed files. {len(pending_files)} remaining.")
            pdf_files = pending_files

        # Set up a directory-level semaphore to process multiple documents concurrently
        sem = asyncio.Semaphore(self.config.vlm_max_concurrent_documents)

        async def process_with_semaphore(index: int, pdf_path: str) -> ProcessResult:
            async with sem:
                self.logger.info(f"[{index + 1}/{len(pdf_files)}] Processing {os.path.basename(pdf_path)} concurrently...")
                return await self._process_async(pdf_path)

        tasks = [process_with_semaphore(i, path) for i, path in enumerate(pdf_files)]
        results = await asyncio.gather(*tasks)
        return results
