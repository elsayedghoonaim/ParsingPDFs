"""Document classification module for the PDF extraction pipeline."""

import fitz  # PyMuPDF
import logging
from . import ClassificationResult, PipelineConfig

logger = logging.getLogger("pdf_extraction")


def _classify_single_page(page, classification_max_image_coverage: float, classification_min_chars_per_page: int) -> dict:
    """Classifies a single PDF page based on text character count and image coverage."""
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return {"is_extractable": False, "chars_found": 0, "image_coverage": 0.0}

    is_scanned = False
    max_coverage = 0.0
    for img in page.get_image_info():
        x0, y0, x1, y1 = img["bbox"]
        coverage = ((x1 - x0) * (y1 - y0)) / page_area
        max_coverage = max(max_coverage, coverage)
        if coverage > classification_max_image_coverage:
            is_scanned = True
            break

    chars = len(page.get_text().strip())
    is_extractable = (not is_scanned) and (chars > classification_min_chars_per_page)
    return {
        "is_extractable": is_extractable,
        "chars_found": chars,
        "image_coverage": round(max_coverage, 4),
    }


def classify_document(pdf_bytes: bytes, config: PipelineConfig) -> ClassificationResult:
    """Classifies a PDF document as extractable or scanned using page sampling and majority vote."""
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        if page_count == 0:
            return ClassificationResult(
                doc_type="scanned",
                confidence_metrics={
                    "chars_found": [],
                    "image_coverage": [],
                    "pages_sampled": [],
                    "page_count": 0
                }
            )

        # Determine which pages to sample
        if config.classification_strategy == "first_page":
            sample_indices = [0]
        elif config.classification_strategy == "sample":
            if page_count == 1:
                sample_indices = [0]
            elif page_count == 2:
                sample_indices = [0, 1]
            else:
                sample_indices = [0, page_count // 2, page_count - 1]
        else:
            # Default fallback to sample
            if page_count == 1:
                sample_indices = [0]
            elif page_count == 2:
                sample_indices = [0, 1]
            else:
                sample_indices = [0, page_count // 2, page_count - 1]

        results = []
        for index in sample_indices:
            page = doc[index]
            res = _classify_single_page(page, config.classification_max_image_coverage, config.classification_min_chars_per_page)
            results.append(res)

        extractable_count = sum(1 for r in results if r["is_extractable"])
        doc_type = "extractable" if extractable_count > len(sample_indices) / 2 else "scanned"

        confidence_metrics = {
            "chars_found": [r["chars_found"] for r in results],
            "image_coverage": [r["image_coverage"] for r in results],
            "pages_sampled": sample_indices,
            "page_count": page_count
        }

        logger.info(f"Document classified as {doc_type.upper()} ({page_count} pages, strategy={config.classification_strategy})")
        logger.debug(
            f"Classification detail: sampled={sample_indices}, "
            f"chars={confidence_metrics['chars_found']}, "
            f"coverage={confidence_metrics['image_coverage']}"
        )

        return ClassificationResult(doc_type=doc_type, confidence_metrics=confidence_metrics)

    except Exception as e:
        logger.error(f"Error during document classification: {e}", exc_info=True)
        return ClassificationResult(
            doc_type="scanned",
            confidence_metrics={
                "chars_found": [],
                "image_coverage": [],
                "pages_sampled": [],
                "page_count": 0,
                "error": str(e)
            }
        )
    finally:
        if doc is not None:
            doc.close()
