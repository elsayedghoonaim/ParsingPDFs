"""Unit tests for the document classifier module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_extraction.core.classifier import classify_document, _classify_single_page
from pdf_extraction.config import load_config


def make_config(**kwargs):
    return load_config(**kwargs)


def make_minimal_text_pdf() -> bytes:
    """
    Generate a minimal valid PDF with a single text-heavy page using PyMuPDF.
    Returns the PDF as bytes.
    """
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    # Insert a large block of text to simulate an extractable page
    text = "This is test content. " * 100
    page.insert_text((50, 50), text, fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_minimal_empty_pdf() -> bytes:
    """
    Returns bytes that look like a truncated/invalid PDF to exercise the classifier's
    error-handling path (which returns 'scanned' on any fitz failure).
    """
    # fitz.open() raises on truly invalid bytes, causing classify_document to return 'scanned'
    return b"%PDF-1.4 invalid truncated"


class TestClassifyDocument:
    def test_text_pdf_classified_as_extractable(self):
        pdf_bytes = make_minimal_text_pdf()
        config = make_config(classification_min_chars_per_page=10)  # low threshold
        result = classify_document(pdf_bytes, config)
        assert result.doc_type == "extractable"

    def test_empty_pdf_classified_as_scanned(self):
        pdf_bytes = make_minimal_empty_pdf()
        config = make_config()
        result = classify_document(pdf_bytes, config)
        assert result.doc_type == "scanned"
        assert result.confidence_metrics["page_count"] == 0

    def test_result_has_confidence_metrics(self):
        pdf_bytes = make_minimal_text_pdf()
        config = make_config(classification_min_chars_per_page=10)
        result = classify_document(pdf_bytes, config)
        assert "chars_found" in result.confidence_metrics
        assert "image_coverage" in result.confidence_metrics
        assert "pages_sampled" in result.confidence_metrics
        assert "page_count" in result.confidence_metrics

    def test_first_page_strategy(self):
        pdf_bytes = make_minimal_text_pdf()
        config = make_config(
            classification_strategy="first_page",
            classification_min_chars_per_page=10
        )
        result = classify_document(pdf_bytes, config)
        assert result.confidence_metrics["pages_sampled"] == [0]

    def test_sample_strategy_samples_multiple_pages(self):
        """For a single-page doc, sample strategy still returns just page 0."""
        pdf_bytes = make_minimal_text_pdf()
        config = make_config(classification_strategy="sample")
        result = classify_document(pdf_bytes, config)
        assert 0 in result.confidence_metrics["pages_sampled"]

    def test_high_min_chars_classifies_as_scanned(self):
        """If min_chars threshold is very high, text page will be classified as scanned."""
        pdf_bytes = make_minimal_text_pdf()
        config = make_config(classification_min_chars_per_page=99999)
        result = classify_document(pdf_bytes, config)
        assert result.doc_type == "scanned"

    def test_invalid_pdf_bytes_returns_scanned(self):
        """Invalid PDF bytes should not crash; falls back to scanned."""
        result = classify_document(b"not a valid pdf", make_config())
        assert result.doc_type == "scanned"


class TestClassifySinglePage:
    def test_page_with_enough_text_is_extractable(self):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        # Use real words — fitz only counts characters that survive rendering
        long_text = ("The quick brown fox jumps over the lazy dog. " * 20)
        page.insert_text((50, 50), long_text, fontsize=10)
        # fitz requires save+reload for get_text() to reflect inserted content
        pdf_bytes = doc.tobytes()
        doc.close()
        doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
        result = _classify_single_page(doc2[0], 0.80, 100)  # Low threshold
        doc2.close()
        assert result["is_extractable"] is True
        assert result["chars_found"] >= 100

    def test_page_with_too_little_text_is_not_extractable(self):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "hi", fontsize=10)
        result = _classify_single_page(page, 0.80, 600)
        doc.close()
        assert result["is_extractable"] is False
