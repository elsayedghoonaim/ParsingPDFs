"""Configuration loader, validation, and defaults for the PDF extraction pipeline."""

import os
import copy
import logging
import yaml
from .core import PipelineConfig

# Mapping from flat PipelineConfig fields to nested YAML paths
FIELD_MAP = {
    # Classification
    "classification_min_chars_per_page": ("classification", "min_chars_per_page"),
    "classification_max_image_coverage": ("classification", "max_image_coverage"),
    "classification_strategy":           ("classification", "strategy"),
    # Provider
    "provider_name":     ("provider", "name"),
    "provider_base_url": ("provider", "base_url"),
    "provider_api_key":  ("provider", "api_key"),
    "provider_model":    ("provider", "model"),
    # Custom provider
    "custom_base_url":       ("provider", "custom", "base_url"),
    "custom_api_key":        ("provider", "custom", "api_key"),
    "custom_model":          ("provider", "custom", "model"),
    "custom_headers":        ("provider", "custom", "headers"),
    "custom_supports_batch": ("provider", "custom", "supports_batch"),
    # VLM
    "vlm_describe_figures":       ("vlm", "describe_figures"),
    "vlm_page_batching":          ("vlm", "page_batching"),
    "vlm_max_pages_per_batch":    ("vlm", "max_pages_per_batch"),
    "vlm_max_concurrent_requests":("vlm", "max_concurrent_requests"),
    "vlm_max_concurrent_documents":("vlm", "max_concurrent_documents"),
    "vlm_timeout_seconds":        ("vlm", "timeout_seconds"),
    "vlm_render_dpi":             ("vlm", "render_dpi"),
    "vlm_optional_prompt":        ("vlm", "optional_prompt"),
    "vlm_max_output_tokens":      ("vlm", "max_output_tokens"),
    # Cost
    "cost_enabled":           ("cost", "enabled"),
    "cost_warn_threshold_usd":("cost", "warn_threshold_usd"),
    "cost_price_per_image":   ("cost", "price_per_image"),
    # Output
    "output_directory":         ("output", "directory"),
    "output_state_directory":   ("output", "state_directory"),
    "output_filename_strategy": ("output", "filename_strategy"),
    # Post-processing
    "postprocessing_enabled":                ("postprocessing", "enabled"),
    "postprocessing_strip_code_wrappers":    ("postprocessing", "strip_code_wrappers"),
    "postprocessing_normalize_whitespace":   ("postprocessing", "normalize_whitespace"),
    "postprocessing_fix_table_alignment":    ("postprocessing", "fix_table_alignment"),
    "postprocessing_normalize_headings":     ("postprocessing", "normalize_headings"),
    # Logging
    "logging_level":                    ("logging", "level"),
    "logging_log_classification_details":("logging", "log_classification_details"),
    "logging_log_file":                 ("logging", "log_file"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges nested dicts, where override values win."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _apply_flat_overrides(config: dict, overrides: dict) -> dict:
    """Maps flat underscore keys (representing dataclass fields) to the nested config structure."""
    for k, v in overrides.items():
        if k in FIELD_MAP:
            path = FIELD_MAP[k]
            current = config
            for part in path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[path[-1]] = v
    return config


def load_config(config_path: str = None, **overrides) -> PipelineConfig:
    """
    Loads default configuration, applies user override config, applies flat overrides,
    resolves API keys from environment variables, validates configuration, and returns a frozen PipelineConfig.
    """
    logger = logging.getLogger("pdf_extraction")

    # 1. Load built-in default_config.yaml
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_config.yaml")
    try:
        with open(default_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Default configuration must be a dictionary")
    except Exception as e:
        logger.error(f"Failed to load default configuration from {default_config_path}: {e}")
        # Initialize a basic fallback structure if default file can't be loaded
        config = {
            "classification": {"min_chars_per_page": 600, "max_image_coverage": 0.80, "strategy": "sample"},
            "provider": {"name": "openai", "base_url": None, "api_key": None, "model": "gpt-4o", "custom": {}},
            "vlm": {
                "describe_figures": False, "page_batching": "single",
                "max_pages_per_batch": 10, "max_concurrent_requests": 5,
                "max_concurrent_documents": 3, "timeout_seconds": 120,
                "render_dpi": 200, "optional_prompt": None, "max_output_tokens": 8192,
            },
            "cost": {"enabled": True, "warn_threshold_usd": 5.00},
            "output": {"directory": "./output/documents", "state_directory": "./output/state", "filename_strategy": "url_hash"},
            "postprocessing": {"enabled": True, "strip_code_wrappers": True, "normalize_whitespace": True, "fix_table_alignment": True, "normalize_headings": True},
            "logging": {"level": "INFO", "log_classification_details": True, "log_file": None},
        }

    # 2. Deep-merge user's config.yaml on top if provided
    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
            if user_config is None:
                user_config = {}
            elif not isinstance(user_config, dict):
                raise ValueError(f"User configuration loaded from {config_path} must be a dictionary, got {type(user_config).__name__}")
            config = _deep_merge(config, user_config)
        except Exception as e:
            logger.error(f"Failed to load user configuration from {config_path}: {e}")
            raise ValueError(f"Failed to load user configuration from {config_path}: {e}")

    # 3. Apply any flat overrides keyword args
    if overrides:
        config = _apply_flat_overrides(config, overrides)

    # 4. API key resolution is pushed down to individual provider classes.

    # 5. Validate values
    strategy = config.get("classification", {}).get("strategy")
    if strategy not in ["first_page", "sample"]:
        raise ValueError(f"Invalid classification strategy '{strategy}'. Must be one of ['first_page', 'sample']")

    prov_name = config.get("provider", {}).get("name")
    if prov_name not in ["openai", "google", "ollama", "vllm", "custom"]:
        raise ValueError(f"Invalid provider name '{prov_name}'. Must be one of ['openai', 'google', 'ollama', 'vllm', 'custom']")

    batching = config.get("vlm", {}).get("page_batching")  # Fixed: was "vlm_page_batching"
    if batching not in ["single", "batch"]:
        raise ValueError(f"Invalid page batching '{batching}'. Must be one of ['single', 'batch']")

    dpi = config.get("vlm", {}).get("render_dpi")  # Fixed: was "vlm_render_dpi"
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError(f"vlm_render_dpi must be a positive integer, got {dpi}")

    concurrent = config.get("vlm", {}).get("max_concurrent_requests")
    if not isinstance(concurrent, int) or concurrent <= 0:
        raise ValueError(f"vlm_max_concurrent_requests must be a positive integer, got {concurrent}")

    concurrent_docs = config.get("vlm", {}).get("max_concurrent_documents", 3)
    if not isinstance(concurrent_docs, int) or concurrent_docs <= 0:
        raise ValueError(f"vlm_max_concurrent_documents must be a positive integer, got {concurrent_docs}")

    fn_strategy = config.get("output", {}).get("filename_strategy")
    if fn_strategy not in ["url_hash", "original"]:
        raise ValueError(f"Invalid filename strategy '{fn_strategy}'. Must be one of ['url_hash', 'original']")

    log_lvl = config.get("logging", {}).get("level")
    if log_lvl not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        raise ValueError(f"Invalid log level '{log_lvl}'. Must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR']")

    max_tokens = config.get("vlm", {}).get("max_output_tokens", 8192)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"vlm_max_output_tokens must be a positive integer, got {max_tokens}")

    # 6. Map nested keys to flat dataclass arguments
    kwargs = {}
    for field_name, path in FIELD_MAP.items():
        val = config
        for part in path:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        kwargs[field_name] = val

    return PipelineConfig(**kwargs)


def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Configures the unified pipeline logger."""
    logger = logging.getLogger("pdf_extraction")

    # Prevent duplicate output if the library is used inside a larger project
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    level = getattr(logging, config.logging_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler — prefix codes keep output visually scannable
    _LEVEL_PREFIX = {
        "DEBUG":    "[DEBUG]",
        "INFO":     "[INFO] ",
        "WARNING":  "[WARN] ",
        "ERROR":    "[ERROR]",
        "CRITICAL": "[CRIT] ",
    }

    class _PrefixFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            prefix = _LEVEL_PREFIX.get(record.levelname, f"[{record.levelname}]")
            return f"{prefix} {record.getMessage()}"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_PrefixFormatter())
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (full format with timestamps for debugging)
    if config.logging_log_file:
        try:
            log_dir = os.path.dirname(config.logging_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(config.logging_log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            )
            file_handler.setLevel(logging.DEBUG)  # Always write full detail to file
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Could not open log file {config.logging_log_file}: {e}")

    return logger
