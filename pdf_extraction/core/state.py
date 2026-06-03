"""Pipeline state manager — manages skip/resume operations using JSON state files."""

import os
import json
import logging
import hashlib
import datetime

logger = logging.getLogger("pdf_extraction")


class PipelineState:
    """Manages the processing state of documents to support pipeline resume/skip features."""

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        try:
            os.makedirs(state_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create state directory '{state_dir}': {e}")

    def _state_path(self, doc_id: str) -> str:
        return os.path.join(self.state_dir, f"{doc_id}.json")

    def _read_state(self, doc_id: str) -> dict | None:
        path = self._state_path(doc_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read state file for doc_id {doc_id} at {path}: {e}")
        return None

    def _write_state(self, doc_id: str, state: dict) -> bool:
        path = self._state_path(doc_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to write state file for doc_id {doc_id} at {path}: {e}")
            return False

    def is_completed(self, doc_id: str) -> bool:
        """Check if a document is already marked as successfully completed."""
        try:
            state = self._read_state(doc_id)
            return state is not None and state.get("status") == "completed"
        except Exception as e:
            logger.error(f"Failed to check completion status for doc_id {doc_id}: {e}")
            return False

    def mark_completed(self, doc_id: str, result: dict):
        """Mark a document as successfully completed and save its metadata."""
        state = {
            "doc_id": doc_id,
            "status": "completed",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **result
        }
        self._write_state(doc_id, state)
        logger.debug(f"State: marked {doc_id} as completed")

    def mark_failed(self, doc_id: str, error: str):
        """Mark a document as failed and save the error message."""
        state = {
            "doc_id": doc_id,
            "status": "failed",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error": error
        }
        self._write_state(doc_id, state)
        logger.debug(f"State: marked {doc_id} as failed")

    def get_pending(self, all_ids: list[str]) -> list[str]:
        """Given a list of document IDs, return only the ones that are pending (not completed)."""
        return [doc_id for doc_id in all_ids if not self.is_completed(doc_id)]

    @staticmethod
    def generate_doc_id(source: str) -> str:
        """Generate a stable SHA-256 doc_id (first 32 chars) from a file path or identifier."""
        try:
            return hashlib.sha256(source.encode("utf-8", errors="ignore")).hexdigest()[:32]
        except Exception as e:
            logger.error(f"Failed to generate stable doc_id for source '{source}': {e}")
            # Safe unique fallback using hash of source
            return hashlib.sha256(str(hash(source)).encode("utf-8")).hexdigest()[:32]
