"""Text extraction module using PyMuPDF4LLM."""

import fitz  # PyMuPDF
import pymupdf4llm
import re
import logging
from . import PageResult

logger = logging.getLogger("pdf_extraction")


def extract_text_markdown(pdf_bytes: bytes) -> list[PageResult]:
    """Extracts Markdown text and figure metadata from a PDF document using PyMuPDF4LLM."""
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        if page_count == 0:
            return []

        # Attempt to extract Markdown page chunks
        try:
            md_text = pymupdf4llm.to_markdown(doc=doc, page_chunks=True)
        except Exception as e:
            logger.warning(
                f"pymupdf4llm.to_markdown with page_chunks=True failed: {e}. "
                "Attempting fallback to full markdown split."
            )
            # Fallback: extract the entire document text and split it
            try:
                full_md = pymupdf4llm.to_markdown(doc=doc)
            except Exception as e_no_kwarg:
                logger.warning(
                    f"pymupdf4llm.to_markdown(doc=doc) failed: {e_no_kwarg}. "
                    "Trying positional doc argument."
                )
                full_md = pymupdf4llm.to_markdown(doc)

            # Split by form feed (\f) or standard Markdown horizontal rules (---)
            # page separators inserted by pymupdf4llm
            chunks = re.split(r'\f', full_md)
            if len(chunks) <= 1:
                chunks = re.split(r'\n---\n', full_md)

            # Align chunks to page_count
            if len(chunks) < page_count:
                chunks.extend([""] * (page_count - len(chunks)))
            elif len(chunks) > page_count:
                # Merge any extra split chunks into the last page chunk
                chunks[page_count - 1] = "\n\n".join(chunks[page_count - 1:])
                chunks = chunks[:page_count]

            md_text = [{"text": chunk} for chunk in chunks]

        # Define the pattern to strip image placeholders
        # e.g., "**==> picture [image_ref] intentionally omitted <==**"
        placeholder_pattern = re.compile(r'\*\*==> picture \[.*?\] intentionally omitted <==\*\*')

        page_results = []
        for page_num, chunk in enumerate(md_text):
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            if not text:
                text = ""

            figures_metadata = []
            page = doc[page_num]
            page_area = page.rect.width * page.rect.height

            if page_area > 0:
                for img_info in page.get_image_info():
                    x0, y0, x1, y1 = img_info["bbox"]
                    img_area = (x1 - x0) * (y1 - y0)
                    coverage = img_area / page_area

                    if coverage > 0.05:  # Filter out decorative/small images (< 5%)
                        xref = img_info.get("xref", 0)
                        try:
                            # Extract raw image bytes if available
                            img_data = doc.extract_image(xref)
                            image_bytes = img_data["image"] if img_data else b""
                        except Exception as e_img:
                            logger.debug(f"Failed to extract image bytes for xref {xref}: {e_img}")
                            image_bytes = b""

                        # Fallback: if raw image bytes are empty/unextractable (e.g., vector drawing),
                        # render the exact bounding box region directly from the page pixmap!
                        if not image_bytes or len(image_bytes) == 0:
                            try:
                                # Render the page region bounded by the figure's bbox
                                # Use a 2.0x zoom factor for crisp high-quality rendering (144 DPI equivalent)
                                zoom = 2.0
                                mat = fitz.Matrix(zoom, zoom)
                                rect = fitz.Rect(x0, y0, x1, y1)
                                if not rect.is_empty:
                                    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                                    image_bytes = pix.tobytes("png")
                                    logger.info(f"Rendered vector/inline figure bounding box {rect} to PNG bytes.")
                            except Exception as e_render:
                                logger.warning(f"Failed to render figure bounding box fallback: {e_render}")
                                image_bytes = b""

                        if image_bytes and len(image_bytes) > 0:
                            figures_metadata.append({
                                "bbox": (x0, y0, x1, y1),
                                "xref": xref,
                                "image_bytes": image_bytes,
                                "description": None
                            })

            # Strip image placeholders from text and clean up whitespace
            clean_text = placeholder_pattern.sub('', text).strip()

            page_results.append(PageResult(
                page_num=page_num,
                markdown_text=clean_text,
                figures_metadata=figures_metadata
            ))

        logger.info(f"Extracted {len(page_results)} pages via pymupdf4llm")
        return page_results

    except Exception as e:
        logger.error(f"Error during text/markdown extraction: {e}", exc_info=True)
        return []
    finally:
        if doc is not None:
            doc.close()
