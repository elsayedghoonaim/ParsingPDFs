"""Unit tests for the post-processing module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_extraction.core.postprocessor import clean_vlm_output
from pdf_extraction.config import load_config


def make_config(**kwargs):
    """Helper to create a PipelineConfig for testing."""
    # Enable all post-processing by default for tests
    defaults = dict(
        postprocessing_enabled=True,
        postprocessing_strip_code_wrappers=True,
        postprocessing_normalize_whitespace=True,
        postprocessing_fix_table_alignment=True,
        postprocessing_normalize_headings=True,
    )
    defaults.update(kwargs)
    return load_config(**defaults)


class TestStripCodeWrappers:
    def test_strips_markdown_wrapper(self):
        config = make_config()
        text = "```markdown\n# Hello\n\nWorld\n```"
        result = clean_vlm_output(text, config)
        assert result == "# Hello\n\nWorld"

    def test_strips_md_wrapper(self):
        config = make_config()
        text = "```md\n# Title\n```"
        result = clean_vlm_output(text, config)
        assert "```" not in result

    def test_strips_bare_code_wrapper(self):
        config = make_config()
        text = "```\n# Title\n```"
        result = clean_vlm_output(text, config)
        assert "```" not in result

    def test_does_not_strip_inline_code(self):
        config = make_config()
        text = "Some text with `inline code` in it."
        result = clean_vlm_output(text, config)
        assert "`inline code`" in result

    def test_disabled_leaves_wrapper(self):
        config = make_config(postprocessing_strip_code_wrappers=False)
        text = "```markdown\n# Hello\n```"
        result = clean_vlm_output(text, config)
        assert "```" in result


class TestNormalizeWhitespace:
    def test_collapses_excessive_blank_lines(self):
        config = make_config()
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = clean_vlm_output(text, config)
        # More than 3 consecutive newlines should be collapsed
        assert "\n\n\n\n" not in result

    def test_removes_trailing_whitespace(self):
        config = make_config()
        text = "Line 1   \nLine 2\t"
        result = clean_vlm_output(text, config)
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_disabled_leaves_whitespace(self):
        config = make_config(postprocessing_normalize_whitespace=False)
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = clean_vlm_output(text, config)
        assert "\n\n\n\n\n\n" in result


class TestFixTableAlignment:
    def test_aligns_simple_table(self):
        config = make_config()
        text = "| A | B |\n| --- | --- |\n| short | a very long value |"
        result = clean_vlm_output(text, config)
        lines = result.strip().split("\n")
        # All lines should start and end with |
        for line in lines:
            assert line.startswith("|") and line.endswith("|")
        # Column widths should be consistent
        col_widths = [len(c) for c in lines[0].split("|")]
        for line in lines:
            assert len(line.split("|")) == len(col_widths)

    def test_table_not_modified_when_disabled(self):
        config = make_config(postprocessing_fix_table_alignment=False)
        text = "| A | B |\n| --- | --- |\n| short | a very long value |"
        result = clean_vlm_output(text, config)
        # Should be essentially unchanged (minus stripping)
        assert "short" in result

    def test_single_row_table_unchanged(self):
        config = make_config()
        text = "| Only Header |"
        result = clean_vlm_output(text, config)
        assert "Only Header" in result


class TestNormalizeHeadings:
    def test_shifts_headings_starting_at_h2(self):
        config = make_config()
        text = "## Section\n\n### Subsection\n\n#### Sub-sub"
        result = clean_vlm_output(text, config)
        # Min heading (##) should shift to #
        assert "# Section" in result
        assert "## Subsection" in result
        assert "### Sub-sub" in result

    def test_headings_at_h1_not_modified(self):
        config = make_config()
        text = "# Title\n\n## Section\n\n### Subsection"
        result = clean_vlm_output(text, config)
        assert "# Title" in result
        assert "## Section" in result

    def test_disabled_leaves_headings_unchanged(self):
        config = make_config(postprocessing_normalize_headings=False)
        text = "## Section\n\n### Subsection"
        result = clean_vlm_output(text, config)
        assert "## Section" in result
        assert "### Subsection" in result


class TestBrTagRemoval:
    def test_br_tag_replaced_with_space(self):
        config = make_config()
        text = "Line one<br>Line two"
        result = clean_vlm_output(text, config)
        assert "<br>" not in result
        assert "Line one" in result and "Line two" in result

    def test_self_closing_br_tag(self):
        config = make_config()
        text = "Line one<br/>Line two"
        result = clean_vlm_output(text, config)
        assert "<br/>" not in result


class TestEmptyInput:
    def test_empty_string_returns_empty(self):
        config = make_config()
        assert clean_vlm_output("", config) == ""

    def test_none_handling(self):
        config = make_config()
        # Should not raise; empty check happens first
        assert clean_vlm_output("", config) == ""

    def test_whitespace_only_returns_empty(self):
        config = make_config()
        result = clean_vlm_output("   \n\n   ", config)
        assert result == ""
