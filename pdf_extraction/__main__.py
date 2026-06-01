"""CLI entry point for the PDF-to-Markdown extraction pipeline."""

import argparse
import sys
import os
import glob
from .pipeline import PDFPipeline


def main():
    parser = argparse.ArgumentParser(
        prog="python -m pdf_extraction",
        description="Universal PDF-to-Markdown extraction pipeline with automated classification."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a single PDF file or a directory containing PDFs."
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to the config.yaml file. Uses built-in defaults if omitted."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing: skip already completed documents using state tracking."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Simulate the extraction run without calling any external VLM/OCR APIs."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Override the output directory defined in the configuration."
    )

    args = parser.parse_args()

    # Dynamic CLI overrides mapped to PipelineConfig fields
    overrides = {}
    if args.output:
        overrides["output_directory"] = args.output

    # Instantiate the pipeline
    try:
        pipeline = PDFPipeline(args.config, **overrides)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    input_path = args.input

    if args.dry_run:
        # Dry Run Simulation Mode
        if os.path.isfile(input_path):
            result = pipeline.dry_run(input_path)
            _print_dry_run_table(result, input_path)
        elif os.path.isdir(input_path):
            pdfs = glob.glob(os.path.join(input_path, "*.pdf"))
            pdfs += glob.glob(os.path.join(input_path, "*.PDF"))
            pdfs = sorted(list(set(pdfs)))
            if not pdfs:
                print(f"No PDFs found in directory: '{input_path}'", file=sys.stderr)
                sys.exit(1)
            
            print(f"\n--- DRY RUN SUMMARY FOR {len(pdfs)} FILES ---")
            for pdf in pdfs:
                result = pipeline.dry_run(pdf)
                _print_dry_run_table(result, pdf)
        else:
            print(f"Error: '{input_path}' is not a valid file or directory.", file=sys.stderr)
            sys.exit(1)
    else:
        # Actual Extraction Execution Mode
        if os.path.isfile(input_path):
            result = pipeline.process(input_path)
            _print_single_result_table(result)
            if result.status == "failed":
                sys.exit(1)
        elif os.path.isdir(input_path):
            results = pipeline.process_directory(input_path, resume=args.resume)
            _print_batch_summary_table(results)
            # Exit with code 1 if all files failed and we had files
            if results and all(r.status == "failed" for r in results):
                sys.exit(1)
        else:
            print(f"Error: '{input_path}' is not a valid file or directory.", file=sys.stderr)
            sys.exit(1)


def _print_dry_run_table(result, original_path):
    filename = os.path.basename(original_path)
    cost_str = f"${result.estimated_cost:.4f}" if result.estimated_cost is not None else "Free/Unknown"
    
    print("┌──────────────────────────────────────────────────────────┐")
    print("│                     DRY RUN ANALYSIS                     │")
    print("├─────────────────────────┬────────────────────────────────┤")
    print(f"│ Document File           │ {filename[:30].ljust(30)} │")
    print(f"│ Document Hash ID        │ {result.doc_id.ljust(30)} │")
    print(f"│ Classified Type         │ {result.doc_type.upper().ljust(30)} │")
    print(f"│ Total Pages             │ {str(result.total_pages).ljust(30)} │")
    print(f"│ Pages Needing VLM/OCR   │ {str(result.pages_needing_vlm).ljust(30)} │")
    print(f"│ Figures Detected        │ {str(result.figures_detected).ljust(30)} │")
    print(f"│ Est. VLM API Calls      │ {str(result.estimated_api_calls).ljust(30)} │")
    print(f"│ Est. Cost (USD)         │ {cost_str.ljust(30)} │")
    print("└─────────────────────────┴────────────────────────────────┘")


def _print_single_result_table(result):
    status_emoji = "✅" if result.status == "completed" else "❌" if result.status == "failed" else "⏭️"
    status_text = f"{result.status.upper()} ({status_emoji})"
    doc_type = result.doc_type.upper() if result.doc_type else "UNKNOWN"
    filename = os.path.basename(result.source_path) if result.source_path else "Memory Bytes"
    cost_str = f"${result.estimated_cost:.4f}" if result.estimated_cost is not None else "Free/Unknown"
    
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│                   EXTRACTION RESULT                      │")
    print("├─────────────────────────┬────────────────────────────────┤")
    print(f"│ Document File           │ {filename[:30].ljust(30)} │")
    print(f"│ Status                  │ {status_text.ljust(30)} │")
    print(f"│ Document Type           │ {doc_type.ljust(30)} │")
    print(f"│ Pages Processed         │ {str(result.pages_processed).ljust(30)} │")
    print(f"│ VLM API Calls Made      │ {str(result.vlm_calls_made).ljust(30)} │")
    print(f"│ Execution Time          │ {f'{result.processing_time:.2f} seconds'.ljust(30)} │")
    print(f"│ Estimated Cost          │ {cost_str.ljust(30)} │")
    
    if result.output_path:
        out_rel = os.path.relpath(result.output_path) if os.path.isabs(result.output_path) else result.output_path
        # Truncate output path if too long
        if len(out_rel) > 30:
            out_rel = "..." + out_rel[-27:]
        print(f"│ Output File             │ {out_rel.ljust(30)} │")
    else:
        print("│ Output File             │ None                           │")
        
    print("└─────────────────────────┴────────────────────────────────┘")
    
    if result.error:
        print(f"\n⚠️  Error details: {result.error}\n")


def _print_batch_summary_table(results):
    if not results:
        print("\n[INFO] No documents were processed in this run.")
        return

    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    total_time = sum(r.processing_time for r in results)
    total_vlm = sum(r.vlm_calls_made for r in results)
    
    total_cost = sum(r.estimated_cost for r in results if r.estimated_cost)
    cost_str = f"${total_cost:.4f}" if total_cost > 0 else "Free/Unknown"
    
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│                     BATCH RUN SUMMARY                    │")
    print("├─────────────────────────┬────────────────────────────────┤")
    print(f"│ Total Documents         │ {str(len(results)).ljust(30)} │")
    print(f"│ Completed (✅)          │ {str(completed).ljust(30)} │")
    print(f"│ Failed (❌)             │ {str(failed).ljust(30)} │")
    print(f"│ Skipped (⏭️)             │ {str(skipped).ljust(30)} │")
    print(f"│ Total Execution Time    │ {f'{total_time:.2f} seconds'.ljust(30)} │")
    print(f"│ Total VLM API Calls     │ {str(total_vlm).ljust(30)} │")
    print(f"│ Cumulative Cost (USD)   │ {cost_str.ljust(30)} │")
    print("└─────────────────────────┴────────────────────────────────┘")

    if failed > 0:
        print("\n❌ FAILED DOCUMENTS:")
        for r in results:
            if r.status == "failed":
                filename = os.path.basename(r.source_path) if r.source_path else r.doc_id
                print(f"  • {filename}: {r.error}")
        print()


if __name__ == "__main__":
    main()
