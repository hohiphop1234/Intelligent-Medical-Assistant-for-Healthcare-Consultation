from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

from config import CHUNK_OVERLAP, CHUNK_SIZE, CRAWL_RATE_LIMIT, CRAWL_WHITELIST
from src.utils import chunk_words, stable_hash

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    requests = None
    BeautifulSoup = None


class WebCrawler:
    """On-demand crawler limited to trusted medical domains."""

    SEARCH_URLS = [
        "https://html.duckduckgo.com/html/?q={query}+site:tamanhhospital.vn+OR+site:vinmec.com+OR+site:hellobacsi.com+OR+site:moh.gov.vn",
    ]

    def __init__(self):
        self.cache: dict[str, str] = {}
        self.last_request = 0.0

    def search(self, query: str, entities: list[str]) -> list[dict[str, Any]]:
        if requests is None or BeautifulSoup is None:
            return []
        search_terms = entities or [query]
        chunks: list[dict[str, Any]] = []
        for term in search_terms[:3]:
            for url in self._search_links(term)[:3]:
                content = self._fetch_page(url)
                if content:
                    chunks.extend(self._chunk_content(content, url=url, entity=term))
        return chunks

    def _search_links(self, query: str) -> list[str]:
        links: list[str] = []
        for template in self.SEARCH_URLS:
            search_url = template.format(query=quote_plus(query))
            html = self._get(search_url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                if "uddg=" in href:
                    from urllib.parse import parse_qs, unquote
                    parsed = urlparse(href)
                    qs = parse_qs(parsed.query)
                    if "uddg" in qs:
                        href = unquote(qs["uddg"][0])
                href = urljoin(search_url, href)
                if self._allowed(href) and not self._is_search_result_url(href) and href not in links:
                    links.append(href)
        return links

    def _fetch_page(self, url: str) -> str | None:
        if self._is_search_result_url(url):
            return None
        html = self._get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body
        if not main:
            return None
        text = main.get_text(separator=" ", strip=True)
        normalized = text.lower()
        if "search results for:" in normalized or "no drug package labels found" in normalized:
            return None
        return text

    def _get(self, url: str) -> str | None:
        if not self._allowed(url):
            return None
        if url in self.cache:
            return self.cache[url]
        self._rate_limit()
        try:
            response = requests.get(
                url,
                timeout=12,
                headers={"User-Agent": "MedicalRAG/1.0 educational prototype"},
            )
            if response.status_code != 200:
                return None
            self.cache[url] = response.text
            return response.text
        except Exception:
            return None

    def _chunk_content(self, content: str, url: str, entity: str) -> list[dict[str, Any]]:
        chunks = []
        for index, text in enumerate(chunk_words(content, CHUNK_SIZE, CHUNK_OVERLAP), 1):
            if len(text.split()) < 50:
                continue
            chunks.append(
                {
                    "id": f"crawl-{stable_hash(url + str(index), 20)}",
                    "content": text,
                    "source": urlparse(url).netloc,
                    "entity": entity,
                    "url": url,
                    "is_crawled": True,
                    "metadata": {
                        "source": urlparse(url).netloc,
                        "entity": entity,
                        "url": url,
                        "title": "Crawled trusted source",
                        "section": "",
                        "risk_level": "medium",
                        "category": "crawled",
                    },
                }
            )
        return chunks

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(host == domain or host.endswith("." + domain) for domain in CRAWL_WHITELIST)

    def _is_search_result_url(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        return (
            "search.cfm" in path
            or "query-meta" in path
            or "search" in path and "query=" in query
        )

    def _rate_limit(self) -> None:
        min_interval = 1.0 / max(CRAWL_RATE_LIMIT, 0.1)
        elapsed = time.time() - self.last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request = time.time()
