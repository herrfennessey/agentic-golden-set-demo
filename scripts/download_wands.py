#!/usr/bin/env python3
"""
Download the WANDS dataset from the official Wayfair GitHub repository.

WANDS (Wayfair ANnotation Dataset) contains:
- 42,994 products
- 480 queries
- 233,448 relevance judgments

Source: https://github.com/wayfair/WANDS
"""

import sys
from pathlib import Path

import httpx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from goldendemo.config import settings

WANDS_BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"

FILES = [
    "product.csv",
    "query.csv",
    "label.csv",
]


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination."""
    print(f"Downloading {url}...")

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

        dest.write_bytes(response.content)
        print(f"  Saved to {dest} ({len(response.content):,} bytes)")


def main() -> None:
    """Download all WANDS dataset files."""
    print("=" * 60)
    print("WANDS Dataset Downloader")
    print("=" * 60)
    print()

    # Create data directory
    settings.wands_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {settings.wands_dir}")
    print()

    for filename in FILES:
        url = f"{WANDS_BASE_URL}/{filename}"
        dest = settings.wands_dir / filename

        if dest.exists():
            print(f"Skipping {filename} (already exists)")
            continue

        try:
            download_file(url, dest)
        except httpx.HTTPError as e:
            print(f"ERROR downloading {filename}: {e}")
            sys.exit(1)

    print()
    print("=" * 60)
    print("Download complete!")
    print()
    print("Dataset files:")
    for filename in FILES:
        filepath = settings.wands_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  - {filename}: {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
