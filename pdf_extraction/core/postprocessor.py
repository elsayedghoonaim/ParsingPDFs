"""Post-processing module — cleans and normalizes raw VLM/OCR text output."""

import re
import logging
from . import PipelineConfig

logger = logging.getLogger("pdf_extraction")


def clean_vlm_output(text: str, config: PipelineConfig) -> str:
    """
    Apply post-processing cleanup steps to the VLM/OCR output sequentially.
    Each step can be individually enabled/disabled via the config.
    """
    if not text:
        return ""

    # Step 1: Strip outer markdown code wrappers
    if config.postprocessing_strip_code_wrappers:
        text_stripped = text.strip()
        if text_stripped.startswith("```") and text_stripped.endswith("```"):
            text_stripped = re.sub(r'^```(?:markdown|md)?\s*\n', '', text_stripped)
            text_stripped = re.sub(r'\n```\s*$', '', text_stripped)
            text = text_stripped

    # Step 2: Normalize whitespace
    if config.postprocessing_normalize_whitespace:
        # Collapse 4 or more consecutive blank lines (which means 3+ empty lines) into 2
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # Remove trailing whitespace from each line
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # Step 3: Fix markdown table alignment
    if config.postprocessing_fix_table_alignment:
        def _realign_table(match):
            """Re-pad a Markdown table so columns align."""
            lines = match.group(0).strip().split('\n')
            if len(lines) < 2:
                return match.group(0)
            
            # Parse rows into cells
            rows = []
            for line in lines:
                # Strip outer pipes and split by pipe character
                cells = [c.strip() for c in line.strip('|').split('|')]
                rows.append(cells)
            
            # Find max width for each column
            num_cols = max(len(row) for row in rows)
            col_widths = [0] * num_cols
            for row in rows:
                for i, cell in enumerate(row):
                    if i < num_cols:
                        # Skip the separator row (containing only - and :) for width calculation
                        if not re.match(r'^[-:]+$', cell):
                            col_widths[i] = max(col_widths[i], len(cell))
            
            # Rebuild the table with padding
            result_lines = []
            for row_idx, row in enumerate(rows):
                padded = []
                for i in range(num_cols):
                    cell = row[i] if i < len(row) else ''
                    if row_idx == 1:
                        # Separator row - make sure it's at least 3 chars long
                        padded.append('-' * max(col_widths[i], 3))
                    else:
                        padded.append(cell.ljust(col_widths[i]))
                result_lines.append('| ' + ' | '.join(padded) + ' |')
            
            return '\n'.join(result_lines)

        # Match consecutive lines starting and ending with |
        text = re.sub(
            r'(?:^\|.+\|$\n?)+',
            _realign_table,
            text,
            flags=re.MULTILINE
        )

    # Step 4: Normalize headings
    if config.postprocessing_normalize_headings:
        # Find all headings at the start of any line
        heading_matches = re.findall(r'^(#{1,6})\s', text, flags=re.MULTILINE)
        if heading_matches:
            # Determine the minimum heading level used in the text
            min_level = min(len(h) for h in heading_matches)
            if min_level > 1:
                # Shift all headings up so the smallest heading becomes level 1 (#)
                shift = min_level - 1
                def _shift_heading(m):
                    hashes = m.group(1)
                    new_level = max(1, len(hashes) - shift)
                    return '#' * new_level + ' '
                text = re.sub(r'^(#{1,6})\s', _shift_heading, text, flags=re.MULTILINE)

    # Step 5: Clean up pymupdf4llm markers and HTML tags
    # Remove picture start/end boundary comments and any following br tags
    text = re.sub(r'\*+\s*-----\s*Start of picture text\s*-----\s*\*+\s*(?:<br\s*/?>)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*+\s*-----\s*End of picture text\s*-----\s*\*+\s*(?:<br\s*/?>)?', '', text, flags=re.IGNORECASE)
    
    # Replace raw <br> tags left inside text lines with normal spaces
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    
    # Strip common legal disclaimers/attribution notice banners (like Google's paper reprint grant)
    disclaimers = [
        r'Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works\s*[\.\s]*'
    ]
    for pattern in disclaimers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Strip double-tilde formatting artifacts in equations (e.g. ~~_√_~~ -> √)
    text = re.sub(r'~~_(.*?)_~~', r'\1', text)
    
    # Clean up empty formatting or spacing leftovers
    text = re.sub(r'\[\s*\]', '', text)
    
    # Collapse double spaces inside paragraphs
    text = re.sub(r'(?<!\n)  +', ' ', text)

    return text.strip()
