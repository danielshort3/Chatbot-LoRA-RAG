"""Async web crawler for collecting VGJ content."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import random
import re
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import aiohttp
import bs4
import trafilatura
import requests
from tqdm.auto import tqdm

# settings
BASE_URL = "https://www.visitgrandjunction.com"
ADDITIONAL_DOMAINS = ["campaign-archive.com", "mailchi.mp"]
DATA_DIR_TXT = Path("data/html_txt")
RAW_HTML_DIR = Path("data/raw_html")
NO_TEXT_HTML_DIR = Path("data/html_no_text")
MIME_DIR = Path("data/mime")
HASH_RECORDS = Path("data/hashes.json")
for d in (DATA_DIR_TXT, RAW_HTML_DIR, NO_TEXT_HTML_DIR, MIME_DIR):
    d.mkdir(parents=True, exist_ok=True)

CRAWL_DELAY = 0.5
N_WORKERS = 10
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
    "Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]

HASH_DB = json.loads(HASH_RECORDS.read_text()) if HASH_RECORDS.exists() else {}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def upsert_hash(url: str, h: str) -> None:
    HASH_DB[url] = h
    HASH_RECORDS.write_text(json.dumps(HASH_DB))


class RateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self.next_ts = 0.0
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        async with self.lock:
            await asyncio.sleep(max(0.0, self.next_ts - time.time()))
            self.next_ts = time.time() + self.delay

    async def __aexit__(self, *_exc: object) -> None:
        return None


def robots_disallow(domain: str) -> list[str]:
    try:
        txt = requests.get(f"https://{domain}/robots.txt", timeout=10).text.lower()
        return [
            l.split(":", 1)[1].strip()
            for l in txt.splitlines()
            if l.startswith("disallow")
        ]
    except Exception:
        return []


def internal_set(base: str, extra: list[str]) -> set[str]:
    return {urlparse(base).netloc, *extra}


def allowed(url: str, nets: set[str], dis_map: dict[str, list[str]]) -> bool:
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        return False
    if not any(p.netloc == n or p.netloc.endswith("." + n) for n in nets):
        return False
    return not any(url.startswith(path) for path in dis_map.get(p.netloc, []))


async def sitemap_seed(base: str, nets: set[str]) -> list[str]:
    try:
        r = requests.get(f"{base}/sitemap.xml", timeout=15)
        r.raise_for_status()
        locs = re.findall(r"<loc>(.*?)</loc>", r.text)
        return [u for u in locs if allowed(u, nets, {})]
    except Exception:
        return []


async def fetch(
    session: aiohttp.ClientSession, url: str, rl: RateLimiter
) -> tuple[str | None, bytes]:
    for attempt in range(MAX_RETRIES):
        try:
            async with rl:
                async with session.get(url, timeout=20) as r:
                    r.raise_for_status()
                    mime = r.headers.get("content-type", "text/html").split(";")[0]
                    return mime, await r.read()
        except Exception:
            await asyncio.sleep(BACKOFF_FACTOR * attempt)
    return None, b""


async def worker(
    name: str,
    idx: int,
    session: aiohttp.ClientSession,
    q: asyncio.Queue[str],
    seen: set[str],
    delay: float,
    nets: set[str],
    dis: dict[str, list[str]],
) -> None:
    rl = RateLimiter(delay)
    bar = tqdm(total=0, position=idx + 1, desc=name, unit="pg", leave=True)
    while True:
        url = await q.get()
        q.task_done()
        if url in seen:
            continue
        seen.add(url)
        bar.set_description(f"{name} {url}")
        mime, body = await fetch(session, url, rl)
        if not body:
            bar.update()
            continue
        sha = sha256(body)
        if HASH_DB.get(url) == sha:
            bar.update()
            continue
        upsert_hash(url, sha)
        uid = hashlib.md5(url.encode()).hexdigest()
        if mime != "text/html":
            ext = mimetypes.guess_extension(mime) or ".bin"
            (MIME_DIR / f"{uid}{ext}").write_bytes(body)
            bar.update()
            continue
        soup = bs4.BeautifulSoup(body, "lxml")
        text = trafilatura.extract(body) or ""
        unsupported = "your browser is not supported for this experience" in text.lower()
        if len(text) < 100 or unsupported:
            (NO_TEXT_HTML_DIR / f"{uid}.html").write_bytes(body)
            bar.update()
            continue
        (RAW_HTML_DIR / f"{uid}.html").write_bytes(body)
        (DATA_DIR_TXT / f"{uid}.txt").write_text(text)
        (DATA_DIR_TXT / f"{uid}.url").write_text(url)
        for a in soup.find_all("a", href=True):
            link, _ = urldefrag(urljoin(url, a["href"]))
            if allowed(link, nets, dis):
                q.put_nowait(link)
        bar.update()


async def crawl(seed: list[str]) -> None:
    if not seed:
        raise ValueError("Seed list empty – nothing to crawl.")
    nets = internal_set(BASE_URL, ADDITIONAL_DOMAINS)
    dis = {n: robots_disallow(n) for n in nets}
    q: asyncio.Queue[str] = asyncio.Queue()
    [q.put_nowait(u) for u in seed]
    seen: set[str] = set()
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            asyncio.create_task(
                worker(f"w{i}", i, session, q, seen, CRAWL_DELAY, nets, dis)
            )
            for i in range(N_WORKERS)
        ]
        await q.join()
        for t in tasks:
            t.cancel()
