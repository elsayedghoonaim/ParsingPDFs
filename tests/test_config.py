"""Unit tests for config loading, FIELD_MAP, and validation."""

import pytest
import os
import sys
import tempfile
import yaml

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_extraction.config import load_config, FIELD_MAP, _deep_merge, _apply_flat_overrides
from pdf_extraction.core import PipelineConfig


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"vlm": {"render_dpi": 200, "timeout_seconds": 120}}
        override = {"vlm": {"render_dpi": 300}}
        result = _deep_merge(base, override)
        assert result["vlm"]["render_dpi"] == 300
        assert result["vlm"]["timeout_seconds"] == 120  # untouched

    def test_does_not_mutate_base(self):
        base = {"vlm": {"render_dpi": 200}}
        override = {"vlm": {"render_dpi": 300}}
        _deep_merge(base, override)
        assert base["vlm"]["render_dpi"] == 200  # original untouched

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result["b"] == 2

    def test_none_value_override(self):
        base = {"a": "some_value"}
        override = {"a": None}
        result = _deep_merge(base, override)
        assert result["a"] is None


class TestApplyFlatOverrides:
    def test_provider_name_override(self):
        config = {"provider": {"name": "openai"}}
        result = _apply_flat_overrides(config, {"provider_name": "google"})
        assert result["provider"]["name"] == "google"

    def test_vlm_render_dpi_override(self):
        config = {"vlm": {"render_dpi": 200}}
        result = _apply_flat_overrides(config, {"vlm_render_dpi": 300})
        assert result["vlm"]["render_dpi"] == 300

    def test_output_directory_override(self):
        config = {"output": {"directory": "./old"}}
        result = _apply_flat_overrides(config, {"output_directory": "./new"})
        assert result["output"]["directory"] == "./new"

    def test_unknown_key_ignored(self):
        config = {"vlm": {"render_dpi": 200}}
        result = _apply_flat_overrides(config, {"nonexistent_key": 999})
        assert result == {"vlm": {"render_dpi": 200}}  # unchanged


class TestFieldMap:
    """Verify FIELD_MAP keys exactly match PipelineConfig field names."""

    def test_all_field_map_keys_exist_in_pipeline_config(self):
        config_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
        missing = [k for k in FIELD_MAP if k not in config_fields]
        assert missing == [], f"FIELD_MAP keys not in PipelineConfig: {missing}"

    def test_readme_kwargs_in_field_map(self):
        """Verify that the kwargs documented in the README are all present in FIELD_MAP."""
        documented_kwargs = [
            "classification_min_chars_per_page",
            "classification_max_image_coverage",
            "classification_strategy",
            "provider_name",
            "provider_model",
            "provider_base_url",
            "provider_api_key",
            "vlm_describe_figures",
            "vlm_page_batching",
            "vlm_max_pages_per_batch",
            "vlm_max_concurrent_requests",
            "vlm_max_concurrent_documents",
            "vlm_timeout_seconds",
            "vlm_render_dpi",
            "vlm_optional_prompt",
            "vlm_max_output_tokens",
            "cost_enabled",
            "cost_warn_threshold_usd",
            "cost_price_per_image",
            "output_directory",
            "output_state_directory",
            "output_filename_strategy",
            "logging_level",
            "logging_log_classification_details",
            "logging_log_file",
            "postprocessing_enabled",
            "postprocessing_strip_code_wrappers",
            "postprocessing_normalize_whitespace",
            "postprocessing_fix_table_alignment",
            "postprocessing_normalize_headings",
        ]
        missing = [k for k in documented_kwargs if k not in FIELD_MAP]
        assert missing == [], f"Documented kwargs not in FIELD_MAP: {missing}"


class TestLoadConfig:
    def test_loads_defaults_when_no_config_provided(self):
        config = load_config()
        assert isinstance(config, PipelineConfig)
        assert config.classification_strategy == "sample"
        assert config.provider_name == "openai"
        assert config.vlm_render_dpi == 200
        assert config.vlm_max_output_tokens == 8192

    def test_kwarg_overrides_work(self):
        config = load_config(vlm_render_dpi=300, provider_name="google", logging_level="DEBUG")
        assert config.vlm_render_dpi == 300
        assert config.provider_name == "google"
        assert config.logging_level == "DEBUG"

    def test_output_directory_kwarg(self):
        config = load_config(output_directory="/tmp/test_output")
        assert config.output_directory == "/tmp/test_output"

    def test_vlm_concurrency_kwarg(self):
        config = load_config(vlm_max_concurrent_requests=10, vlm_max_concurrent_documents=5)
        assert config.vlm_max_concurrent_requests == 10
        assert config.vlm_max_concurrent_documents == 5

    def test_yaml_file_override(self, tmp_path):
        user_cfg = {
            "classification": {"min_chars_per_page": 1000},
            "provider": {"name": "google", "model": "gemini-2.0-flash"},
            "vlm": {"render_dpi": 250},
        }
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump(user_cfg))
        config = load_config(str(cfg_file))
        assert config.classification_min_chars_per_page == 1000
        assert config.provider_name == "google"
        assert config.provider_model == "gemini-2.0-flash"
        assert config.vlm_render_dpi == 250

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="classification strategy"):
            load_config(classification_strategy="invalid")

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="provider name"):
            load_config(provider_name="anthropic")

    def test_invalid_batching_raises(self):
        with pytest.raises(ValueError, match="page batching"):
            load_config(vlm_page_batching="streaming")

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log level"):
            load_config(logging_level="VERBOSE")

    def test_invalid_filename_strategy_raises(self):
        with pytest.raises(ValueError, match="filename strategy"):
            load_config(output_filename_strategy="timestamp")

    def test_empty_yaml_file_uses_defaults(self, tmp_path):
        """An empty YAML file should not crash and should fall back to defaults."""
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        config = load_config(str(cfg_file))
        assert config.classification_strategy == "sample"

    def test_kwarg_overrides_yaml(self, tmp_path):
        """Kwargs should take precedence over YAML file settings."""
        user_cfg = {"vlm": {"render_dpi": 150}}
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(user_cfg))
        config = load_config(str(cfg_file), vlm_render_dpi=400)
        assert config.vlm_render_dpi == 400
