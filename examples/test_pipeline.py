#!/usr/bin/env python3
"""
Sanity test script for the Universal PDF-to-Markdown Extraction Pipeline.
Downloads a sample PDF and runs both the dry-run analysis and full extraction.
"""

import asyncio
import os
import sys

# Ensure the script's directory is in the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from download import download_pdf
    from pdf_extraction import PDFPipeline
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you have installed all dependencies via requirements.txt:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# A reliable public PDF URL (Attention Is All You Need paper)
TEST_PDF_URL = "https://source.z2data.com/2017/4/16/4/16/55/206/1559202880/860-005-213R004.pdf"
DOWNLOAD_DIR = "./downloaded_test_pdfs"


async def main():
    print("=" * 60)
    print("PDF-TO-MARKDOWN EXTRACTION PIPELINE TESTER")
    print("=" * 60)
    
    # 1. Download the PDF
    print(f"\nStep 1: Downloading test PDF from:\n{TEST_PDF_URL}")
    print("Streaming download in progress...")
    try:
        download_result = await download_pdf(TEST_PDF_URL, output_dir=DOWNLOAD_DIR)
        if not download_result.success:
            print(f"❌ Download failed: {download_result.error}")
            sys.exit(1)
        print(f"✅ Download successful! Saved to: {download_result.local_path}")
        pdf_path = download_result.local_path
    except Exception as e:
        print(f"❌ Failed to download PDF: {e}")
        sys.exit(1)

    # 2. Initialize the Pipeline
    print("\nStep 2: Initializing the PDFPipeline...")
    config_path = "config.yaml"
    
    # Check if a custom config exists, fallback to default if not
    if not os.path.exists(config_path):
        config_path = None
        print("💡 User config.yaml not found, falling back to default library configuration.")
    else:
        print(f"💡 Loading active configuration from: {config_path}")

    try:
        pipeline = PDFPipeline(config_path=config_path)
        print("✅ Pipeline initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        sys.exit(1)

    # 3. Execute Dry Run Simulation
    print("\nStep 3: Executing Dry-Run Simulation (No API calls)...")
    try:
        dry_run_result = pipeline.dry_run(pdf_path)
        print("-" * 40)
        print(f"📄 Document ID:       {dry_run_result.doc_id}")
        print(f"📊 Document Type:     {dry_run_result.doc_type.upper()}")
        print(f"📖 Total Pages:       {dry_run_result.total_pages}")
        print(f"🖼️ Detected Figures:   {dry_run_result.figures_detected}")
        print(f"🤖 Estimated VLM Calls: {dry_run_result.estimated_api_calls}")
        if dry_run_result.estimated_cost is not None:
            print(f"💰 Estimated Cost:    ${dry_run_result.estimated_cost:.4f} USD")
        else:
            print("💰 Estimated Cost:    Unknown (Local execution / custom provider)")
        print("-" * 40)
    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        sys.exit(1)

    # 4. Execute Full Extraction
    print("\nStep 4: Running Full PDF Extraction Pipeline...")
    print("This will auto-classify the document and generate the formatted markdown...")
    
    # Reset/clear the cached completed state for this specific test document
    # so that the test runner always performs a fresh full extraction.
    try:
        state_file = os.path.join(pipeline.state.state_dir, f"{dry_run_result.doc_id}.json")
        if os.path.exists(state_file):
            os.remove(state_file)
            print("💡 Cleared cached completion state for a fresh test run.")
    except Exception as state_err:
        print(f"💡 Note: Could not reset cached state: {state_err}")
        
    try:
        result = pipeline.process(pdf_path)
        
        print("\n" + "=" * 60)
        print("Pipeline Execution Completed!")
        print("=" * 60)
        print(f"📄 Status:           {result.status.upper()}")
        if result.status == "completed":
            print(f"📁 Source Document:  {result.source_path}")
            print(f"📝 Output Markdown:  {result.output_path}")
            print(f"📊 Extracted Type:   {result.doc_type.upper()}")
            print(f"📖 Pages Processed:  {result.pages_processed}")
            print(f"🤖 VLM Calls Made:   {result.vlm_calls_made}")
            if result.estimated_cost is not None:
                print(f"💰 Total API Cost:   ${result.estimated_cost:.4f} USD")
            print(f"⚡ Processing Time:  {result.processing_time:.2f} seconds")
            

        else:
            print(f"❌ Error Detail: {result.error}")
            
    except Exception as e:
        print(f"❌ Full pipeline execution encountered an error: {e}")


if __name__ == "__main__":
    # Check selector loop policy if running on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
