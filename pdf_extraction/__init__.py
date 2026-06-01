"""
PDF Extraction Pipeline — Universal PDF-to-Markdown converter.

Usage:
    from pdf_extraction import PDFPipeline

    pipeline = PDFPipeline("config.yaml")
    result = pipeline.process("document.pdf")
"""

from .pipeline import PDFPipeline
from .core import ProcessResult, DryRunResult, ClassificationResult, PageResult

__all__ = [
    "PDFPipeline",
    "ProcessResult",
    "DryRunResult",
    "ClassificationResult",
    "PageResult",
]
