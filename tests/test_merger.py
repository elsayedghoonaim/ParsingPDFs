"""Unit tests for the document merger module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_extraction.core.merger import merge_to_document
from pdf_extraction.core import PageResult


def make_page(num: int, text: str, descriptions: list[str] = None) -> PageResult:
    pr = PageResult(page_num=num, markdown_text=text)
    if descriptions:
        pr.figure_descriptions = descriptions
    return pr


class TestMergeScanned:
    def test_single_page(self):
        pages = [make_page(0, "# Hello\n\nWorld")]
        result = merge_to_document(pages, "scanned")
        assert "# Hello" in result
        assert "World" in result

    def test_multiple_pages_joined(self):
        pages = [make_page(0, "Page one"), make_page(1, "Page two")]
        result = merge_to_document(pages, "scanned")
        assert "Page one" in result
        assert "Page two" in result

    def test_pages_sorted_by_number(self):
        pages = [make_page(1, "Page two"), make_page(0, "Page one")]
        result = merge_to_document(pages, "scanned")
        assert result.index("Page one") < result.index("Page two")

    def test_empty_pages_skipped(self):
        pages = [make_page(0, "Content"), make_page(1, ""), make_page(2, "More")]
        result = merge_to_document(pages, "scanned")
        assert "Content" in result
        assert "More" in result

    def test_returns_stripped_text(self):
        pages = [make_page(0, "  Content  ")]
        result = merge_to_document(pages, "scanned")
        assert result == result.strip()


class TestMergeExtractable:
    def test_text_only_pages(self):
        pages = [make_page(0, "# Title"), make_page(1, "## Section")]
        result = merge_to_document(pages, "extractable")
        assert "# Title" in result
        assert "## Section" in result

    def test_figure_descriptions_appended(self):
        pages = [make_page(0, "Some text", descriptions=["A bar chart showing quarterly revenue"])]
        result = merge_to_document(pages, "extractable")
        assert "Some text" in result
        assert "[Figure]" in result
        assert "bar chart" in result

    def test_figure_description_formatted_as_blockquote(self):
        pages = [make_page(0, "Text", descriptions=["A pie chart"])]
        result = merge_to_document(pages, "extractable")
        assert "> **[Figure]**:" in result

    def test_multiple_figures_on_page(self):
        pages = [make_page(0, "Text", descriptions=["Figure 1", "Figure 2"])]
        result = merge_to_document(pages, "extractable")
        assert "Figure 1" in result
        assert "Figure 2" in result

    def test_page_with_only_figure_no_text(self):
        pages = [make_page(0, "", descriptions=["A diagram"])]
        result = merge_to_document(pages, "extractable")
        assert "A diagram" in result

    def test_empty_input_returns_empty(self):
        result = merge_to_document([], "extractable")
        assert result == ""

    def test_all_empty_pages_returns_empty(self):
        pages = [make_page(0, ""), make_page(1, "  ")]
        result = merge_to_document(pages, "extractable")
        assert result == ""


class TestMergeFinalDocument:
    def test_no_leading_trailing_newlines(self):
        pages = [make_page(0, "Content")]
        result = merge_to_document(pages, "scanned")
        assert not result.startswith("\n")
        assert not result.endswith("\n")

    def test_pages_separated_by_double_newlines(self):
        pages = [make_page(0, "Page one"), make_page(1, "Page two")]
        result = merge_to_document(pages, "scanned")
        assert "\n\n" in result
