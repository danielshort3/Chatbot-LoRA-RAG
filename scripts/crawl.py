from vgj_chat.data.crawl import crawl, sitemap_seed, BASE_URL, internal_set, ADDITIONAL_DOMAINS
import asyncio

if __name__ == "__main__":
    seed = asyncio.run(sitemap_seed(BASE_URL, internal_set(BASE_URL, ADDITIONAL_DOMAINS)))
    asyncio.run(crawl(seed))
