"""Standalone PDF downloader utility.

Supports:
- Single URL and batch URL downloads
- Extracting URLs using regex from a text file
- Connection throttling / concurrency control via Semaphore
- Download streaming to memory then writing to disk
- Custom User-Agent headers
- Retry with exponential backoff and jitter for transient errors (429, 5xx)
- Size verification using Content-Length headers
- PDF signature validation (first 4 bytes match b"%PDF")
- Skippable downloads for existing files (unless --force is specified)
"""

import asyncio
import aiohttp
import argparse
import os
import sys
import re
import hashlib
import time
import random
import urllib.parse
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("pdf_downloader")

@dataclass
class DownloadResult:
    url: str
    success: bool
    local_path: str | None
    filename: str | None
    file_size: int
    error: str | None

# Configuration constants
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_CONCURRENCY = 10
DEFAULT_OUTPUT_DIR = "./pdfs"
CHUNK_SIZE = 8192  # For streaming downloads
PDF_MAGIC_BYTES = b"%PDF"

def _extract_filename(url: str, response: aiohttp.ClientResponse) -> str:
    """Determine the filename from Content-Disposition header, URL path, or URL hash.
    
    Priority:
    1. Content-Disposition header's filename parameter
    2. URL path basename
    3. MD5 hash of URL
    """
    filename = None
    cd = response.headers.get("Content-Disposition")
    if cd:
        matches = re.findall(r'filename[^;=\n]*=([\"\']?)(.+?)\1(;|$)', cd)
        if matches:
            filename = matches[0][1]

    if not filename:
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        if path:
            filename = path.split('/')[-1]

    if not filename:
        filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"

    # Strip query parameters if they somehow sneaked into URL path extraction
    if "?" in filename:
        filename = filename.split("?")[0]

    filename = urllib.parse.unquote(filename).strip()

    if not filename or '.' not in filename:
        filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"

    content_type = response.headers.get("Content-Type", "")
    if not filename.lower().endswith(".pdf") and "application/pdf" in content_type.lower():
        filename += ".pdf"

    # Sanitize: remove any characters not in [a-zA-Z0-9._-], replace with _
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Ensure it's not completely empty or just underscores
    if not filename.strip('_'):
        filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"
        
    return filename

async def _validate_pdf(data: bytes) -> bool:
    """Validate that the first 4 bytes match %PDF."""
    return data[:4] == PDF_MAGIC_BYTES

async def download_pdf(
    url: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> DownloadResult:
    """Download a single PDF with streaming, size checks, signature validation, and retries."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge custom headers on top of default User-Agent
    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        req_headers.update(headers)

    for attempt in range(max_retries):
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.get(url, headers=req_headers, allow_redirects=True) as response:
                    # Check status for transient retryable issues
                    if response.status in [429, 502, 503, 504]:
                        sleep = (2 ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(f"Status {response.status} for URL: {url}. Retrying in {sleep:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(sleep)
                        continue

                    response.raise_for_status()

                    # Check Content-Type
                    content_type = response.headers.get("Content-Type", "")
                    if "html" in content_type.lower():
                        logger.warning(f"URL returned HTML instead of PDF: {url}")
                        return DownloadResult(
                            url=url,
                            success=False,
                            local_path=None,
                            filename=None,
                            file_size=0,
                            error="Response is HTML, not PDF"
                        )

                    # Extract filename and construct local path
                    filename = _extract_filename(url, response)
                    local_path = os.path.join(output_dir, filename)

                    # Skip download if file exists and force is not requested
                    if os.path.exists(local_path) and not force:
                        file_size = os.path.getsize(local_path)
                        logger.info(f"Skipping (exists): {filename}")
                        return DownloadResult(
                            url=url,
                            success=True,
                            local_path=local_path,
                            filename=filename,
                            file_size=file_size,
                            error=None
                        )

                    # Stream download into bytearray
                    data = bytearray()
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        data.extend(chunk)

                    # Validate PDF signature
                    if not await _validate_pdf(bytes(data)):
                        logger.warning(f"Signature mismatch for {url}: expected first 4 bytes to match %PDF")
                        return DownloadResult(
                            url=url,
                            success=False,
                            local_path=None,
                            filename=filename,
                            file_size=len(data),
                            error="Downloaded file is not a valid PDF"
                        )

                    # Verify Content-Length if provided
                    expected_size = response.headers.get("Content-Length")
                    if expected_size:
                        try:
                            expected_len = int(expected_size)
                            if expected_len != len(data):
                                logger.warning(f"Size mismatch for {url}: expected {expected_len} bytes, got {len(data)} bytes.")
                                if attempt < max_retries - 1:
                                    sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                                    logger.warning(f"Retrying size mismatch in {sleep_time:.1f}s...")
                                    await asyncio.sleep(sleep_time)
                                    continue
                        except ValueError:
                            pass

                    # Write to file
                    with open(local_path, "wb") as f:
                        f.write(data)

                    logger.info(f"Downloaded: {filename} ({len(data)} bytes)")
                    return DownloadResult(
                        url=url,
                        success=True,
                        local_path=local_path,
                        filename=filename,
                        file_size=len(data),
                        error=None
                    )

        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                logger.error(f"Download timed out for URL: {url} (all retries exhausted)")
                return DownloadResult(
                    url=url,
                    success=False,
                    local_path=None,
                    filename=None,
                    file_size=0,
                    error="Download timed out"
                )
            sleep = (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(f"Timeout on {url}. Retrying in {sleep:.1f}s...")
            await asyncio.sleep(sleep)

        except aiohttp.ClientError as e:
            if attempt == max_retries - 1:
                logger.error(f"Client error for URL {url}: {e} (all retries exhausted)")
                return DownloadResult(
                    url=url,
                    success=False,
                    local_path=None,
                    filename=None,
                    file_size=0,
                    error=f"Download error: {e}"
                )
            sleep = (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(f"Client error on {url}: {e}. Retrying in {sleep:.1f}s...")
            await asyncio.sleep(sleep)

        except Exception as e:
            logger.error(f"Unexpected error for URL {url}: {e}")
            return DownloadResult(
                url=url,
                success=False,
                local_path=None,
                filename=None,
                file_size=0,
                error=f"Unexpected error: {e}"
            )

    return DownloadResult(
        url=url,
        success=False,
        local_path=None,
        filename=None,
        file_size=0,
        error="All retry attempts failed"
    )

async def download_batch(
    urls: list[str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    concurrency: int = DEFAULT_CONCURRENCY,
    **kwargs
) -> list[DownloadResult]:
    """Download a batch of URLs with concurrency control."""
    sem = asyncio.Semaphore(concurrency)

    async def _download_with_sem(url: str):
        async with sem:
            return await download_pdf(url, output_dir, **kwargs)

    tasks = [_download_with_sem(url) for url in urls]
    return await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description="Standalone robust PDF downloader utility.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Single PDF URL to download")
    group.add_argument("--urls-file", help="Path to a text file containing URLs")

    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="Directory to save downloaded PDFs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files even if they already exist")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent downloads in batch mode")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-download request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Maximum number of retry attempts for transient errors")
    parser.add_argument("--header", action="append", help="Custom HTTP headers in format 'Name: Value' (can be specified multiple times)")

    args = parser.parse_args()

    # Parse headers
    custom_headers = {}
    if args.header:
        for h in args.header:
            if ":" not in h:
                print(f"Warning: Skipping invalid header format: '{h}'. Use 'Name: Value'.", file=sys.stderr)
                continue
            key, _, value = h.partition(":")
            custom_headers[key.strip()] = value.strip()

    # Configure event loop policy for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.url:
        # Download single URL
        result = asyncio.run(download_pdf(
            url=args.url,
            output_dir=args.output,
            force=args.force,
            headers=custom_headers or None,
            timeout=args.timeout,
            max_retries=args.max_retries
        ))
        
        if result.success:
            print(f"\n✅ Downloaded: {result.local_path} ({result.file_size} bytes)")
        else:
            print(f"\n❌ Failed: {result.error}")
            sys.exit(1)
    else:
        # Download batch from file
        if not os.path.exists(args.urls_file):
            print(f"Error: URLs file '{args.urls_file}' does not exist.", file=sys.stderr)
            sys.exit(1)

        try:
            with open(args.urls_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading URLs file: {e}", file=sys.stderr)
            sys.exit(1)

        # Find URLs using regex
        raw_urls = re.findall(r'https?://[^\s]+', content)
        
        # Clean URLs (strip trailing quotes, brackets, and punctuation commonly matched by basic regex)
        urls = []
        for url in raw_urls:
            cleaned = url.rstrip('.,;)"\' ]>}')
            if cleaned:
                urls.append(cleaned)

        if not urls:
            print("No URLs found in the specified file.", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(urls)} URLs. Initiating batch download...")
        start_time = time.time()
        
        results = asyncio.run(download_batch(
            urls=urls,
            output_dir=args.output,
            concurrency=args.concurrency,
            force=args.force,
            headers=custom_headers or None,
            timeout=args.timeout,
            max_retries=args.max_retries
        ))

        duration = time.time() - start_time
        success = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        print(f"\n{'='*50}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*50}")
        print(f"Total processed : {len(results)}")
        print(f"Succeeded       : {success}")
        print(f"Failed          : {failed}")
        print(f"Time taken      : {duration:.2f}s")
        print(f"{'='*50}\n")

        if failed > 0:
            print("Failed URLs:")
            for r in results:
                if not r.success:
                    print(f"  ❌ {r.url} - Error: {r.error}")
            sys.exit(1)

if __name__ == "__main__":
    main()
