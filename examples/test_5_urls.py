#!/usr/bin/env python3
"""
Test the Universal PDF-to-Markdown Extraction Pipeline on 5 URLs.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from download import download_pdf
    from pdf_extraction import PDFPipeline
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

URLS = [
    "https://arxiv.org/pdf/1706.03762", # Attention Is All You Need
    "https://arxiv.org/pdf/2005.11401", # Retrieval-Augmented Generation
    "https://arxiv.org/pdf/2303.08774", # GPT-4 Technical Report
    "https://arxiv.org/pdf/2103.00020", # CLIP
    "https://arxiv.org/pdf/2201.11903"  # Chain-of-Thought Prompting
]

DOWNLOAD_DIR = "./downloaded_test_pdfs"

async def main():
    print("=" * 60)
    print("TESTING 5 URLs FROM urls.txt")
    print("=" * 60)

    config_path = "config.yaml" if os.path.exists("config.yaml") else None
    pipeline = PDFPipeline(config_path=config_path)
    
    results = []

    for i, url in enumerate(URLS, 1):
        print(f"\n--- Processing URL {i}/5 ---")
        print(f"URL: {url}")
        
        # Download
        download_result = await download_pdf(url, output_dir=DOWNLOAD_DIR)
        if not download_result.success:
            print(f"[ERROR] Download failed: {download_result.error}")
            results.append({"url": url, "status": "failed", "error": download_result.error})
            continue
            
        print(f"[SUCCESS] Downloaded to: {download_result.local_path}")
        
        # Process
        try:
            result = pipeline.process(download_result.local_path)
            if result.status == "completed":
                print(f"[SUCCESS] Extraction completed! Output: {result.output_path}")
                results.append({
                    "url": url,
                    "status": "completed",
                    "output": result.output_path,
                    "doc_type": result.doc_type,
                    "pages": result.pages_processed,
                    "time": result.processing_time
                })
            else:
                print(f"[ERROR] Extraction failed: {result.error}")
                results.append({"url": url, "status": "error", "error": result.error})
        except Exception as e:
            print(f"[ERROR] Exception during processing: {e}")
            results.append({"url": url, "status": "exception", "error": str(e)})

    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    for res in results:
        if res["status"] == "completed":
            print(f"[SUCCESS]: {res['url']}")
            print(f"   - Type: {res['doc_type']}, Pages: {res['pages']}, Time: {res['time']:.2f}s")
            print(f"   - Output: {res['output']}")
        else:
            print(f"[ERROR]: {res['url']}")
            print(f"   - Error: {res.get('error')}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
