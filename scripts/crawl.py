import asyncio

from vgj_chat.data.crawl import (
    ADDITIONAL_DOMAINS,
    BASE_URL,
    crawl,
    internal_set,
    sitemap_seed,
)

if __name__ == "__main__":
    seed = asyncio.run(
        sitemap_seed(BASE_URL, internal_set(BASE_URL, ADDITIONAL_DOMAINS))
    )
    asyncio.run(crawl(seed))
