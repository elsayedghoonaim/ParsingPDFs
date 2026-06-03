"""Unit tests for the pipeline state manager."""

import pytest
import os
import sys
import json
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_extraction.core.state import PipelineState


@pytest.fixture
def state(tmp_path):
    """Return a PipelineState backed by a temporary directory."""
    return PipelineState(str(tmp_path / "state"))


class TestStateDirectory:
    def test_creates_state_directory(self, tmp_path):
        state_dir = str(tmp_path / "new_state_dir")
        assert not os.path.exists(state_dir)
        PipelineState(state_dir)
        assert os.path.exists(state_dir)


class TestDocIdGeneration:
    def test_generates_string_id(self):
        doc_id = PipelineState.generate_doc_id("some/path/file.pdf")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_same_input_same_id(self):
        path = "C:/some/path/document.pdf"
        id1 = PipelineState.generate_doc_id(path)
        id2 = PipelineState.generate_doc_id(path)
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = PipelineState.generate_doc_id("file1.pdf")
        id2 = PipelineState.generate_doc_id("file2.pdf")
        assert id1 != id2

    def test_id_is_hex_string(self):
        doc_id = PipelineState.generate_doc_id("test.pdf")
        # SHA-256 hex, truncated to 32 chars
        assert all(c in "0123456789abcdef" for c in doc_id)
        assert len(doc_id) == 32

    def test_empty_string_does_not_raise(self):
        doc_id = PipelineState.generate_doc_id("")
        assert isinstance(doc_id, str)


class TestMarkCompleted:
    def test_marks_completed(self, state):
        doc_id = "test_doc_001"
        state.mark_completed(doc_id, {"output_file": "out.md"})
        assert state.is_completed(doc_id)

    def test_writes_state_file(self, state):
        doc_id = "test_doc_002"
        state.mark_completed(doc_id, {"pages": 5})
        state_file = state._state_path(doc_id)
        assert os.path.exists(state_file)

    def test_state_file_contains_metadata(self, state):
        doc_id = "test_doc_003"
        state.mark_completed(doc_id, {"output_file": "result.md", "pages": 10})
        with open(state._state_path(doc_id)) as f:
            data = json.load(f)
        assert data["status"] == "completed"
        assert data["doc_id"] == doc_id
        assert data["output_file"] == "result.md"
        assert data["pages"] == 10
        assert "timestamp" in data


class TestMarkFailed:
    def test_marks_failed(self, state):
        doc_id = "fail_doc_001"
        state.mark_failed(doc_id, "Connection timeout")
        assert not state.is_completed(doc_id)

    def test_failed_state_contains_error(self, state):
        doc_id = "fail_doc_002"
        state.mark_failed(doc_id, "Connection timeout")
        with open(state._state_path(doc_id)) as f:
            data = json.load(f)
        assert data["status"] == "failed"
        assert data["error"] == "Connection timeout"


class TestIsCompleted:
    def test_returns_false_for_unknown_doc(self, state):
        assert not state.is_completed("nonexistent_doc")

    def test_returns_true_after_mark_completed(self, state):
        doc_id = "completed_doc"
        state.mark_completed(doc_id, {})
        assert state.is_completed(doc_id)

    def test_returns_false_for_failed_doc(self, state):
        doc_id = "failed_doc"
        state.mark_failed(doc_id, "error")
        assert not state.is_completed(doc_id)

    def test_completed_overrides_failed(self, state):
        doc_id = "retry_doc"
        state.mark_failed(doc_id, "first attempt failed")
        state.mark_completed(doc_id, {"output_file": "out.md"})
        assert state.is_completed(doc_id)


class TestGetPending:
    def test_all_pending_when_none_completed(self, state):
        ids = ["a", "b", "c"]
        pending = state.get_pending(ids)
        assert pending == ids

    def test_completed_excluded_from_pending(self, state):
        state.mark_completed("a", {})
        state.mark_completed("c", {})
        pending = state.get_pending(["a", "b", "c"])
        assert pending == ["b"]

    def test_empty_list_returns_empty(self, state):
        assert state.get_pending([]) == []
