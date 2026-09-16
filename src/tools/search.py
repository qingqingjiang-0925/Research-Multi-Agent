"""搜索工具层。

四个后端，按可用性自动降级：
- `TavilyBackend`      ：需要 TAVILY_API_KEY，质量最好，直接返回正文片段。
- `BingBackend`        ：零 key，抓 www.bing.com/search，国内网络可直连，默认兜底。
- `DuckDuckGoBackend`  ：零 key，抓 html.duckduckgo.com，海外网络可用。
- `MockSearchBackend`  ：离线演示 / 单元测试用，不发任何网络请求。

所有联网后端都带**熔断**：连续连接失败即标记 `dead=True`，后续 query 直接跳过，
避免在网络不通的环境里按超时时间静默空转十几分钟。

`SearchToolkit` 是 Search Agent 真正调用的门面：
搜索 -> URL 去重 -> 质量打分过滤 -> 正文抓取 -> 近似去重 -> 截断，一次返回结构化证据。
"""

from __future__ import annotations

import base64
import binascii
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from src.config import Settings, get_settings
from src.schemas import Document, make_doc_id
from src.tools.extractor import DEFAULT_UA, ExtractedPage, WebExtractor
from src.utils.dedup import DedupIndex, normalize
from src.utils.logger import get_logger

log = get_logger("search")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    score: float = 0.0
    raw_content: str = ""
    published: str = ""


class SearchBackend(Protocol):
    name: str
    dead: bool

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


_CONNECT_ERRORS = (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout)


class _CircuitBreaker:
    """连续连接失败即熔断，把"网络不可达"从分钟级空转变成秒级失败。"""

    def __init__(self, name: str, threshold: int = 2):
        self.name = name
        self.threshold = threshold
        self.dead = False
        self._fails = 0

    def note_failure(self, exc: Exception) -> None:
        if not isinstance(exc, _CONNECT_ERRORS):
            return
        self._fails += 1
        if self._fails >= self.threshold and not self.dead:
            self.dead = True
            log.error(
                f"{self.name} 连续 {self._fails} 次连接失败（{type(exc).__name__}），"
                f"判定当前网络不可达，已熔断后续请求"
            )

    def note_success(self) -> None:
        self._fails = 0


# --------------------------------------------------------------------------- #
# Tavily
# --------------------------------------------------------------------------- #
class TavilyBackend:
    """直接打 Tavily REST API，不绑定 SDK 版本。"""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, search_depth: str = "basic", timeout: int = 20):
        self.api_key = api_key
        self.search_depth = search_depth
        self.timeout = timeout
        self._breaker = _CircuitBreaker(self.name)

    @property
    def dead(self) -> bool:
        return self._breaker.dead

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": True,
        }
        try:
            resp = httpx.post(self.endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Tavily 搜索失败({query!r}): {exc}")
            self._breaker.note_failure(exc)
            return []
        self._breaker.note_success()
        out: list[SearchResult] = []
        for item in data.get("results", []) or []:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                    score=float(item.get("score") or 0.0),
                    raw_content=(item.get("raw_content") or "")[:20000],
                    published=str(item.get("published_date") or ""),
                )
            )
        return out


# --------------------------------------------------------------------------- #
# DuckDuckGo (HTML 端点，无需 key)
# --------------------------------------------------------------------------- #
class DuckDuckGoBackend:
    name = "duckduckgo"
    endpoints = (
        "https://html.duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    )

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self._breaker = _CircuitBreaker(self.name)
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": DEFAULT_UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )

    @property
    def dead(self) -> bool:
        return self._breaker.dead

    @staticmethod
    def _unwrap(href: str) -> str:
        """DDG 的链接是 /l/?uddg=<encoded> 跳转，需要还原真实 URL。"""
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if parsed.path in ("/l/", "/l") and parsed.query:
            qs = parse_qs(parsed.query)
            if "uddg" in qs:
                return unquote(qs["uddg"][0])
        return href

    def _parse(self, html: str) -> list[SearchResult]:
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:  # noqa: BLE001
            soup = BeautifulSoup(html, "html.parser")

        results: list[SearchResult] = []
        anchors = soup.select("a.result__a") or soup.select("a.result-link")
        snippets = soup.select(".result__snippet") or soup.select("td.result-snippet")
        for i, a in enumerate(anchors):
            url = self._unwrap(a.get("href", ""))
            if not url.startswith("http"):
                continue
            title = a.get_text(" ", strip=True)
            snippet = snippets[i].get_text(" ", strip=True) if i < len(snippets) else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet, score=0.0))
        return results

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        for endpoint in self.endpoints:
            if self._breaker.dead:
                return []
            for attempt in range(2):
                try:
                    resp = self._client.post(endpoint, data={"q": query, "kl": "wt-wt"})
                    if resp.status_code == 200:
                        parsed = self._parse(resp.text)
                        self._breaker.note_success()
                        if parsed:
                            return parsed[:max_results]
                        break
                    if resp.status_code in (403, 429):
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
                except Exception as exc:  # noqa: BLE001
                    log.debug(f"DDG {endpoint} 失败: {exc}")
                    self._breaker.note_failure(exc)
                    if self._breaker.dead:
                        return []
                    time.sleep(0.8)
        return []


# --------------------------------------------------------------------------- #
# Bing（HTML 端点，无需 key，国内可直连）
# --------------------------------------------------------------------------- #
class BingBackend:
    """抓 www.bing.com/search 的 HTML 结果页。

    选它做默认兜底是因为在国内网络下 DuckDuckGo 的 html/lite 端点均无法建连，
    而 Bing 可直连、无需 API key，且结果页结构（`li.b_algo`）长期稳定。
    """

    name = "bing"
    endpoint = "https://www.bing.com/search"

    def __init__(self, timeout: int = 20, market: str = "zh-CN"):
        self.timeout = timeout
        self.market = market
        self._breaker = _CircuitBreaker(self.name)
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": DEFAULT_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )

    @property
    def dead(self) -> bool:
        return self._breaker.dead

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _unwrap(href: str) -> str:
        """Bing 有时把外链包成 /ck/a?...&u=a1<base64url>，需要还原真实 URL。"""
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if parsed.path.startswith("/ck/") and parsed.query:
            qs = parse_qs(parsed.query)
            blob = (qs.get("u") or [""])[0]
            if blob.startswith("a1"):
                blob = blob[2:]
            try:
                pad = "=" * (-len(blob) % 4)
                return base64.urlsafe_b64decode(blob + pad).decode("utf-8", "ignore")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                return ""
        return href

    def _parse(self, html: str) -> list[SearchResult]:
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:  # noqa: BLE001
            soup = BeautifulSoup(html, "html.parser")

        results: list[SearchResult] = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a") or li.select_one("a")
            if a is None:
                continue
            url = self._unwrap(a.get("href", ""))
            if not url.startswith("http"):
                continue
            if "bing.com" in urlparse(url).netloc:
                continue
            title = a.get_text(" ", strip=True)
            snippet_node = (
                li.select_one(".b_caption p")
                or li.select_one(".b_lineclamp2")
                or li.select_one("p")
            )
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            date_node = li.select_one(".news_dt") or li.select_one("span[tabindex]")
            published = date_node.get_text(" ", strip=True)[:40] if date_node else ""
            if not title:
                continue
            results.append(
                SearchResult(title=title, url=url, snippet=snippet, score=0.0, published=published)
            )
        return results

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        params = {
            "q": query,
            "count": str(max(10, max_results * 2)),
            "mkt": self.market,
            "setlang": self.market.split("-")[0],
            "FORM": "QBLH",
        }
        for attempt in range(2):
            if self._breaker.dead:
                return []
            try:
                resp = self._client.get(self.endpoint, params=params)
            except Exception as exc:  # noqa: BLE001
                log.debug(f"Bing 请求失败({query!r}): {exc}")
                self._breaker.note_failure(exc)
                if self._breaker.dead:
                    return []
                time.sleep(0.8)
                continue
            if resp.status_code in (403, 429):
                log.debug(f"Bing 限流 {resp.status_code}，退避重试")
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                log.debug(f"Bing HTTP {resp.status_code}")
                return []
            self._breaker.note_success()
            return self._parse(resp.text)[:max_results]
        return []


# --------------------------------------------------------------------------- #
# Mock（离线）
# --------------------------------------------------------------------------- #
class MockSearchBackend:
    """不发网络请求，返回可复现的假结果。用于 `main.py demo` 与单元测试。"""

    name = "mock"

    def __init__(self, corpus: Optional[list[dict]] = None):
        self.corpus = corpus or []
        self.dead = False

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        from src.mock import mock_search_results

        if self.corpus:
            picked = [
                SearchResult(**{k: v for k, v in item.items() if k in SearchResult.__annotations__})
                for item in self.corpus
            ]
            return picked[:max_results]
        return mock_search_results(query, max_results)


# --------------------------------------------------------------------------- #
# 质量过滤
# --------------------------------------------------------------------------- #
BLOCKED_DOMAINS = {
    "pinterest.com", "facebook.com", "instagram.com", "tiktok.com", "youtube.com",
    "x.com", "twitter.com", "linkedin.com", "quora.com", "zhihu.com/signin",
    "amazon.com", "taobao.com", "jd.com", "aliexpress.com", "ebay.com",
    "wikipedia.org",  # 二手来源，调研里不作为一手证据
}
TRUSTED_HINTS = (
    ".gov", ".edu", "arxiv.org", "github.com", "github.io", "nature.com", "acm.org",
    "ieee.org", "springer.com", "sciencedirect.com", "openai.com", "anthropic.com",
    "google.com", "deepmind.com", "microsoft.com", "aws.amazon.com", "huggingface.co",
    "langchain.com", "stackoverflow.com", "reuters.com", "bloomberg.com", "36kr.com",
    "infoq.cn", "ossinsight.io", "stackshare.io", "indeed.com", "zhipin.com", "lagou.com",
)
STOPWORDS = {
    "的", "了", "和", "与", "在", "是", "对", "及", "或", "a", "an", "the", "of", "to",
    "in", "for", "and", "or", "on", "with", "is", "are", "what", "how", "why",
}


def _terms(text: str) -> set[str]:
    return {t for t in re.split(r"[\s,，。.;；:：/、()（）\[\]\"']+", normalize(text)) if len(t) >= 2 and t not in STOPWORDS}


def domain_of(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def score_result(result: SearchResult, query: str, question: str = "") -> float:
    """0-1 的启发式质量分。可解释、可单测。"""
    score = 0.30
    domain = domain_of(result.url)

    # 域名可信度
    if any(domain == b or domain.endswith("." + b) or b in domain for b in BLOCKED_DOMAINS):
        score -= 0.35
    if any(h in domain for h in TRUSTED_HINTS):
        score += 0.20

    # 后端给的相关性分（Tavily 0-1）
    if result.score > 0:
        score += 0.25 * min(max(result.score, 0.0), 1.0)

    # 关键词覆盖
    q_terms = _terms(query) | _terms(question)
    if q_terms:
        text_terms = _terms(result.title) | _terms(result.snippet)
        overlap = len(q_terms & text_terms) / len(q_terms)
        score += 0.30 * min(overlap, 1.0)

    # 摘要信息量
    snippet_len = len(result.snippet or "")
    if snippet_len >= 200:
        score += 0.10
    elif snippet_len >= 60:
        score += 0.05
    elif snippet_len < 20 and not result.raw_content:
        score -= 0.10

    # 标题党 / 无效页惩罚
    if re.search(r"(登录|注册|sign in|log in|404|not found|页面不存在|验证码|captcha)", result.title, re.I):
        score -= 0.25

    return round(min(max(score, 0.0), 1.0), 4)


# --------------------------------------------------------------------------- #
# SearchToolkit
# --------------------------------------------------------------------------- #
@dataclass
class SearchToolkit:
    """Search Agent 的工具门面。"""

    backend: SearchBackend
    extractor: Optional[WebExtractor] = None
    settings: Settings = field(default_factory=get_settings)
    dedup: DedupIndex = field(default_factory=lambda: DedupIndex(threshold=0.72))
    seen_urls: set = field(default_factory=set)
    calls: int = 0
    scrapes: int = 0

    # ------------------------------------------------------------------ #
    def search_once(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        max_results = max_results or self.settings.search_max_results
        if getattr(self.backend, "dead", False):
            log.warning(f"搜索后端 {getattr(self.backend, 'name', '?')} 已熔断，跳过 {query!r}")
            return []
        self.calls += 1
        try:
            results = self.backend.search(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"搜索后端异常({query!r}): {exc}")
            return []
        log.debug(f"query={query!r} -> {len(results)} 条")
        return results

    # ------------------------------------------------------------------ #
    def _scrape(self, doc: Document) -> Document:
        if self.extractor is None or not self.settings.scrape_enabled:
            return doc
        self.scrapes += 1
        page: ExtractedPage = self.extractor.fetch(doc.url)
        if page.ok and len(page.text) > len(doc.best_text):
            doc.content = page.text[: self.settings.scrape_max_chars]
            doc.scraped = True
            if not doc.title and page.title:
                doc.title = page.title
            if not doc.published and page.meta.get("published"):
                doc.published = page.meta["published"]
            doc.score = round(min(1.0, doc.score + 0.12), 4)
        elif page.error:
            log.debug(f"抓取失败 {doc.url}: {page.error}")
        return doc

    # ------------------------------------------------------------------ #
    def collect(self, queries: list[str], subtask_id: str, question: str = "",
                limit: Optional[int] = None, scrape: bool = True) -> list[Document]:
        """执行一批 query，返回去重、过滤、排序后的证据。"""
        settings = self.settings
        limit = limit or settings.max_evidence_per_subtask
        candidates: dict[str, Document] = {}

        for query in queries:
            if not query.strip():
                continue
            for res in self.search_once(query, settings.search_max_results):
                if not res.url.startswith("http"):
                    continue
                doc_id = make_doc_id(res.url)
                if doc_id in self.seen_urls:
                    continue
                score = score_result(res, query, question)
                if score < 0.22:
                    continue
                if doc_id in candidates:
                    if score > candidates[doc_id].score:
                        candidates[doc_id].score = score
                        candidates[doc_id].query = query
                    continue
                candidates[doc_id] = Document(
                    id=doc_id,
                    url=res.url,
                    title=res.title or res.url,
                    snippet=res.snippet[:1200],
                    content=(res.raw_content or "")[: settings.scrape_max_chars],
                    source=getattr(self.backend, "name", "search"),
                    query=query,
                    subtask_id=subtask_id,
                    score=score,
                    published=res.published,
                    scraped=bool(res.raw_content),
                )

        if not candidates:
            return []

        # 正文抓取：只对排名靠前的候选抓，控制耗时
        ordered = sorted(candidates.values(), key=lambda d: d.score, reverse=True)
        to_scrape = [d for d in ordered[: limit + 2] if scrape and len(d.best_text) < 600]
        if to_scrape and self.extractor is not None and settings.scrape_enabled:
            with ThreadPoolExecutor(max_workers=max(1, settings.scrape_concurrency)) as pool:
                futures = {pool.submit(self._scrape, d): d for d in to_scrape}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        log.debug(f"抓取任务异常: {exc}")

        # 近似去重 + URL 去重
        kept: list[Document] = []
        for doc in sorted(candidates.values(), key=lambda d: d.score, reverse=True):
            text = doc.best_text
            if len(normalize(text)) < 80:
                continue
            if self.dedup.is_duplicate(text):
                log.debug(f"近似重复，丢弃: {doc.url}")
                continue
            self.dedup.add(doc.id, text)
            self.seen_urls.add(doc.id)
            kept.append(doc)
            if len(kept) >= limit:
                break
        return kept


# --------------------------------------------------------------------------- #
def build_search_backend(settings: Optional[Settings] = None) -> SearchBackend:
    settings = settings or get_settings()
    name = settings.resolved_search_backend()
    if name == "mock":
        return MockSearchBackend()
    if name == "tavily" and settings.has_tavily:
        return TavilyBackend(settings.tavily_api_key, settings.search_depth, settings.search_timeout)
    if name == "tavily":
        log.warning("未配置 TAVILY_API_KEY，自动降级到 Bing HTML 后端")
    if name == "bing":
        return BingBackend(settings.search_timeout, settings.bing_market)
    return DuckDuckGoBackend(settings.search_timeout)


def build_toolkit(settings: Optional[Settings] = None,
                  backend: Optional[SearchBackend] = None) -> SearchToolkit:
    settings = settings or get_settings()
    backend = backend or build_search_backend(settings)
    # mock 后端自带正文，且 URL 不可解析，禁用抓取
    if getattr(backend, "name", "") == "mock":
        return SearchToolkit(backend=backend, extractor=None, settings=settings)
    extractor = WebExtractor(settings.scrape_timeout, settings.scrape_max_chars) if settings.scrape_enabled else None
    return SearchToolkit(backend=backend, extractor=extractor, settings=settings)
