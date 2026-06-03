#!/usr/bin/env python3
"""
PDF-to-Markdown Extraction Pipeline — Test Runner
Downloads a sample PDF and runs classification, dry-run, and full extraction.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from download import download_pdf
    from pdf_extraction import PDFPipeline
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("  → Run:  pip install -r requirements.txt\n")
    sys.exit(1)

TEST_PDF_URL = "https://source.z2data.com/2017/4/16/4/16/55/206/1559202880/860-005-213R004.pdf"
DOWNLOAD_DIR = "./downloaded_test_pdfs"

# ── ANSI colours (disabled automatically on Windows without colour support) ──
_USE_COLOR = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED    = lambda t: _c("31", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


def _header(title: str) -> None:
    bar = "─" * 56
    print(f"\n{CYAN(bar)}")
    print(f"  {BOLD(title)}")
    print(f"{CYAN(bar)}")


def _ok(msg: str)   -> None: print(f"  {GREEN('✓')} {msg}")
def _warn(msg: str) -> None: print(f"  {YELLOW('⚠')} {msg}")
def _fail(msg: str) -> None: print(f"  {RED('✗')} {msg}")
def _row(label: str, value: str) -> None:
    print(f"  {DIM(label.ljust(22))} {value}")


async def main() -> None:
    _header("PDF-to-Markdown Extraction Pipeline")

    # ── Step 1: Download ────────────────────────────────────────────────────
    print(f"\n{BOLD('Step 1 — Download')}")
    print(f"  URL: {DIM(TEST_PDF_URL)}")
    try:
        t0 = time.time()
        result = await download_pdf(TEST_PDF_URL, output_dir=DOWNLOAD_DIR)
        elapsed = time.time() - t0
        if not result.success:
            _fail(f"Download failed: {result.error}")
            sys.exit(1)
        _ok(f"Downloaded in {elapsed:.1f}s  →  {result.local_path}")
        pdf_path = result.local_path
    except Exception as e:
        _fail(f"Download error: {e}")
        sys.exit(1)

    # ── Step 2: Initialise Pipeline ─────────────────────────────────────────
    print(f"\n{BOLD('Step 2 — Initialise Pipeline')}")
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    config_path = None
    for _c_path in ["config.yaml", os.path.join(_project_root, "config.yaml")]:
        if os.path.exists(_c_path):
            config_path = os.path.abspath(_c_path)
            break

    if config_path:
        _ok(f"Config  →  {config_path}")
    else:
        _warn("No config.yaml found — using built-in defaults (provider: openai)")

    try:
        pipeline = PDFPipeline(config_path=config_path)
        _ok(f"Provider: {pipeline.config.provider_name} / {pipeline.config.provider_model}")
    except Exception as e:
        _fail(f"Pipeline init failed: {e}")
        sys.exit(1)

    # ── Step 3: Dry Run ─────────────────────────────────────────────────────
    print(f"\n{BOLD('Step 3 — Dry Run Analysis')}  {DIM('(no API calls)')}")
    try:
        dry = pipeline.dry_run(pdf_path)
        _row("Document ID:",   dry.doc_id[:16] + "…")
        _row("Type:",          dry.doc_type.upper())
        _row("Pages:",         str(dry.total_pages))
        _row("Figures found:", str(dry.figures_detected))
        _row("Est. API calls:", str(dry.estimated_api_calls))
        if dry.estimated_cost is not None:
            _row("Est. cost:", f"${dry.estimated_cost:.4f} USD")
        else:
            _row("Est. cost:", "N/A (local / custom provider)")
    except Exception as e:
        _fail(f"Dry run error: {e}")
        sys.exit(1)

    # ── Step 4: Full Extraction ─────────────────────────────────────────────
    print(f"\n{BOLD('Step 4 — Full Extraction')}")

    # Clear cached state so we always do a fresh run in tests
    try:
        state_file = os.path.join(pipeline.state.state_dir, f"{dry.doc_id}.json")
        if os.path.exists(state_file):
            os.remove(state_file)
    except Exception:
        pass  # Not critical

    print(f"  Extracting {dry.total_pages} page(s) using {pipeline.config.provider_name}…")

    try:
        t0 = time.time()
        result = pipeline.process(pdf_path)
        elapsed = time.time() - t0

        if result.status == "completed":
            _ok(f"Done in {elapsed:.1f}s")
            print()
            _row("Output file:",    result.output_path)
            _row("Document type:",  result.doc_type.upper())
            _row("Pages extracted:", str(result.pages_processed))
            _row("API calls made:", str(result.vlm_calls_made))
            if result.estimated_cost is not None:
                _row("API cost:", f"${result.estimated_cost:.4f} USD")
            _row("Processing time:", f"{result.processing_time:.1f}s")
        else:
            error = result.error or "Unknown error"

            # Rate-limit / quota error
            if "429" in error or "quota" in error.lower() or "rate limit" in error.lower():
                _fail("API quota exceeded")
                print()
                print(f"  {YELLOW('The API key has hit its free-tier daily limit.')}")
                print(f"  {YELLOW('Options:')}")
                print(f"    • Wait until midnight (Pacific Time) for the quota to reset")
                print(f"    • Enable billing on your Google AI Studio account")
                print(f"    • Use a different API key")
            # Config / key missing error
            elif "configuration error" in error.lower() or "api key" in error.lower():
                _fail("Provider not configured")
                print()
                print(f"  {YELLOW('Missing or invalid API key. Edit config.yaml:')}")
                print(f"    provider:")
                print(f"      name:    google")
                print(f"      api_key: <your-key>")
                print(f"      model:   gemini-2.0-flash-lite")
            # Generic failure
            else:
                _fail(f"Extraction failed: {error}")

    except Exception as e:
        _fail(f"Unexpected error: {e}")

    print()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
