"""Document merger module — merges page results into a single clean Markdown document."""

import re
import logging
from . import PageResult

logger = logging.getLogger("pdf_extraction")


def merge_to_document(page_results: list[PageResult], doc_type: str) -> str:
    """
    Merge a list of PageResult objects into a single clean Markdown document.
    
    Sorts page results, appends formatted figure descriptions to the end of
    extractable pages, collapses excess blank lines to at most 2, and strips
    the final document.
    """
    # 1. Sort page results by page number
    sorted_pages = sorted(page_results, key=lambda pr: pr.page_num)
    merged_parts = []
    
    # 2. Extract and format pages
    if doc_type == "scanned":
        for pr in sorted_pages:
            if pr.markdown_text:
                merged_parts.append(pr.markdown_text)
    else:  # doc_type == "extractable"
        for pr in sorted_pages:
            page_text = pr.markdown_text
            
            # If there are figure descriptions, format and append them
            if pr.figure_descriptions:
                figure_blocks = []
                for desc in pr.figure_descriptions:
                    lines = desc.split('\n')
                    blockquote = '> **[Figure]**: ' + lines[0]
                    for line in lines[1:]:
                        blockquote += '\n> ' + line
                    figure_blocks.append(blockquote)
                
                # Append figure descriptions at the end of page text
                if page_text:
                    page_text = page_text + "\n\n" + "\n\n".join(figure_blocks)
                else:
                    page_text = "\n\n".join(figure_blocks)
                    
            if page_text and page_text.strip():
                merged_parts.append(page_text)
                
    # 3. Join all parts with double newlines
    merged_text = "\n\n".join(merged_parts)
    
    # 4. Final cleanup: strip the final text
    final_text = merged_text.strip()
    
    logger.info(f"Merged {len(page_results)} pages into final document ({len(final_text)} chars)")
    return final_text
