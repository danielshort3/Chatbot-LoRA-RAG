import argparse
import asyncio
from urllib.parse import urlparse, urlunparse

from vgj_chat.data.crawl import (
    ADDITIONAL_DOMAINS,
    BASE_URL,
    DATA_DIR_TXT,
    crawl,
    internal_set,
    sitemap_seed,
)

def prefer_https(urls: set[str]) -> set[str]:
    """
    Ensure that, when both http://example.com/foo and https://example.com/foo
    exist, only the https:// version is kept.

    Works by keying on the *netloc + path + params + query + fragment* portion
    of the URL, ignoring the scheme.
    """
    canonical: dict[tuple[str, str, str, str, str], str] = {}
    for u in urls:
        parts = urlparse(u)
        key = (parts.netloc, parts.path, parts.params, parts.query, parts.fragment)

        # Prefer HTTPS if we’ve seen both schemes for the same key
        if key not in canonical or parts.scheme == "https":
            # Re-assemble the URL to keep any normalization consistent
            canonical[key] = urlunparse(parts)

    return set(canonical.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run crawler")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pages to download",
    )
    args = parser.parse_args()

    # 1. Skip crawl if data already exists
    if args.limit is None and any(DATA_DIR_TXT.glob("*.txt")):
        print(f"{DATA_DIR_TXT} contains data; skipping crawl")
        raise SystemExit

    # 2. Build initial seed set
    raw_seed = asyncio.run(
        sitemap_seed(BASE_URL, internal_set(BASE_URL, ADDITIONAL_DOMAINS))
    )

    # 3. Canonicalize → HTTPS-only
    seed = prefer_https(raw_seed)

    # 4. Crawl!
    asyncio.run(crawl(seed, limit=args.limit))
