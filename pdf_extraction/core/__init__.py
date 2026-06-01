"""Core module — shared data contracts for the PDF extraction pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PageResult:
    """Result of processing a single PDF page."""
    page_num: int
    markdown_text: str
    figures_metadata: list[dict] = field(default_factory=list)
    # figures_metadata entries: {"bbox": (x0,y0,x1,y1), "xref": int, "image_bytes": bytes, "description": str|None}
    figure_descriptions: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of classifying a PDF document."""
    doc_type: str  # "extractable" | "scanned"
    confidence_metrics: dict = field(default_factory=dict)
    # confidence_metrics keys: "chars_found" (list[int]), "image_coverage" (list[float]),
    #                          "pages_sampled" (list[int]), "page_count" (int)


@dataclass
class ProcessResult:
    """Final result returned to the user after processing a document."""
    doc_id: str
    source_path: str
    output_path: str
    doc_type: str                          # "extractable" | "scanned"
    classification_strategy: str           # "first_page" | "sample"
    confidence_metrics: dict = field(default_factory=dict)
    pages_processed: int = 0
    vlm_calls_made: int = 0
    estimated_cost: Optional[float] = None
    processing_time: float = 0.0
    status: str = "pending"                # "completed" | "failed" | "skipped"
    error: Optional[str] = None


@dataclass
class DryRunResult:
    """Result of a dry-run analysis (no API calls made)."""
    doc_id: str
    doc_type: str
    total_pages: int = 0
    pages_needing_vlm: int = 0
    estimated_api_calls: int = 0
    estimated_cost: Optional[float] = None
    figures_detected: int = 0


@dataclass
class PipelineConfig:
    """Validated, frozen configuration for the pipeline."""
    # Classification
    classification_min_chars_per_page: int = 600
    classification_max_image_coverage: float = 0.80
    classification_strategy: str = "sample"

    # Provider
    provider_name: str = "openai"
    provider_base_url: Optional[str] = None
    provider_api_key: Optional[str] = None
    provider_model: str = "gpt-4o"

    # Custom provider
    custom_base_url: str = "http://localhost:8000/v1"
    custom_api_key: str = ""
    custom_model: str = "my-model"
    custom_headers: dict = field(default_factory=dict)
    custom_supports_batch: bool = False

    # VLM
    vlm_describe_figures: bool = False
    vlm_page_batching: str = "single"          # "single" | "batch"
    max_pages_per_batch: int = 10
    max_concurrent_requests: int = 5
    max_concurrent_documents: int = 3
    vlm_timeout_seconds: int = 120
    vlm_render_dpi: int = 200
    optional_prompt: Optional[str] = None

    # Cost
    cost_enabled: bool = True
    cost_warn_threshold_usd: float = 5.00
    cost_price_per_image: Optional[float] = None

    # Output
    output_directory: str = "./output/documents"
    state_directory: str = "./output/state"
    filename_strategy: str = "url_hash"    # "url_hash" | "original"

    # Post-processing
    postprocessing_enabled: bool = True
    postprocessing_strip_code_wrappers: bool = True
    postprocessing_normalize_whitespace: bool = True
    postprocessing_fix_table_alignment: bool = True
    postprocessing_normalize_headings: bool = True

    # Logging
    log_level: str = "INFO"
    log_classification_details: bool = True
    log_file: Optional[str] = None
