"""网页正文提取工具（自研，不依赖 trafilatura / readability）。

流程：下载 -> 去噪 -> 主内容容器定位（文本密度打分）-> 段落级清洗 -> 截断。

自己实现而不是直接调库，是因为调研场景里"提取质量"直接决定后面 Writer 的证据质量，
需要能针对导航页 / 聚合页 /  cookie 弹窗做定制过滤。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from src.utils.logger import get_logger

log = get_logger("extract")

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

NOISE_TAGS = {
    "script", "style", "noscript", "template", "svg", "iframe", "form", "button",
    "input", "select", "textarea", "nav", "footer", "header", "aside", "figure",
    "figcaption", "picture", "video", "audio", "canvas", "map", "object", "embed",
    "dialog", "portal", "marquee", "blink",
}

NOISE_ATTR = re.compile(
    r"(nav|navbar|menu|menubar|sidebar|side-bar|footer|foot|header|head-|breadcrumb|"
    r"comment|share|social|advert|banner|cookie|consent|popup|modal|related|recommend|"
    r"promo|sponsor|subscribe|newsletter|login|signin|signup|register|pagination|pager|"
    r"toc|toolbar|widget|copyright|footer-|adsbygoogle|breadcrumb|tag-list|author-box)",
    re.I,
)

CANDIDATE_SELECTORS = [
    "article",
    "main",
    "[role=main]",
    ".post-content",
    ".entry-content",
    ".article-content",
    ".markdown-body",
    "#content",
    ".content",
    "#main-content",
    ".main-content",
    ".post",
    ".article",
    "#article",
]

BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5",
    "h6", "blockquote", "pre", "td", "th", "tr", "table", "dd", "dt", "br", "hr",
}

_WS = re.compile(r"[ \t\u3000]+")
_MULTI_NL = re.compile(r"\n{3,}")
_MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024

# 主机级熔断：同一域名连续超时/拒连就不再试，避免每个子任务都白等 scrape_timeout 秒。
# 进程内共享（多个 searcher 分支各持一个 toolkit，但网络可达性是全局事实）。
_HOST_FAILS: dict[str, int] = {}
_HOST_BLOCKED: set[str] = set()
_HOST_LOCK = threading.Lock()
_HOST_FAIL_THRESHOLD = 2


@dataclass
class ExtractedPage:
    url: str
    title: str = ""
    text: str = ""
    ok: bool = False
    error: str = ""
    status_code: int = 0
    char_count: int = 0
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# HTML -> 正文
# --------------------------------------------------------------------------- #
def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(list(NOISE_TAGS)):
        tag.decompose()
    # find_all 会先把整棵树的标签物化成列表；decompose 掉父节点后，
    # 列表里剩下的子节点已失效（bs4 会把 attrs 置为 None），必须跳过，否则 AttributeError。
    for tag in soup.find_all(True):
        if not isinstance(tag, Tag) or tag.attrs is None or getattr(tag, "decomposed", False):
            continue
        attrs = " ".join(
            " ".join(v) if isinstance(v, list) else str(v)
            for v in (tag.get("class") or []) + ([tag["id"]] if tag.get("id") else [])
        )
        if not attrs:
            continue
        if NOISE_ATTR.search(attrs):
            text_len = len(tag.get_text(" ", strip=True))
            # 大块内容即使 class 名可疑也保留，避免误杀
            if text_len < 600:
                tag.decompose()


def _visible_text(tag: Tag) -> str:
    return tag.get_text(" ", strip=True)


def _density(tag: Tag) -> float:
    text = _visible_text(tag)
    markup_len = len(str(tag))
    if not text:
        return 0.0
    link_len = sum(len(a.get_text(" ", strip=True)) for a in tag.find_all("a"))
    link_ratio = link_len / max(1, len(text))
    raw = len(text) / max(1.0, markup_len) ** 0.5
    return raw * (1.0 - 0.7 * min(link_ratio, 1.0))


def _pick_container(soup: BeautifulSoup) -> Tag:
    best: Tag | None = None
    best_score = 0.0
    for selector in CANDIDATE_SELECTORS:
        for tag in soup.select(selector)[:3]:
            if not isinstance(tag, Tag):
                continue
            text_len = len(_visible_text(tag))
            if text_len < 200:
                continue
            paragraphs = len(tag.find_all("p"))
            score = _density(tag) * (1 + min(paragraphs, 30) / 30.0)
            if score > best_score:
                best, best_score = tag, score
    if best is not None:
        return best

    body = soup.body or soup
    for tag in body.find_all(["div", "section", "article"], recursive=True):
        text_len = len(_visible_text(tag))
        if text_len < 400:
            continue
        score = _density(tag)
        if score > best_score:
            best, best_score = tag, score
    return best or body


def _tag_to_lines(tag: Tag) -> list[str]:
    lines: list[str] = []

    def walk(node) -> None:
        if isinstance(node, str):
            return
        if not isinstance(node, Tag):
            return
        name = node.name
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _visible_text(node)
            if text:
                level = int(name[1])
                lines.append("\n" + "#" * min(level + 1, 6) + " " + text + "\n")
            return
        if name == "li":
            text = _visible_text(node)
            if text:
                lines.append("- " + text)
            return
        if name in ("pre", "code"):
            text = node.get_text("\n", strip=False).strip("\n")
            if text:
                lines.append("```\n" + text[:1200] + "\n```")
            return
        if name in ("td", "th"):
            text = _visible_text(node)
            if text:
                lines.append("| " + text + " ")
            return
        if name == "br":
            lines.append("\n")
            return
        children = [c for c in node.children if isinstance(c, Tag) and c.name in BLOCK_TAGS]
        if not children:
            text = _visible_text(node)
            if text:
                lines.append(text)
            return
        for child in node.children:
            walk(child)

    walk(tag)
    return lines


_LINE_NOISE = re.compile(r"^(-\s)?(首页|登录|注册|下载|更多|详情|返回|分享|收藏|评论|举报|广告|hot|new|\d+)$", re.I)


def _clean_lines(lines: list[str], min_len: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = _WS.sub(" ", raw).strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("```") or line.startswith("|") or line.startswith("-"):
            pass
        elif len(line) < min_len and not re.search(r"[。.!！?？:：;；]", line):
            continue
        if _LINE_NOISE.match(line):
            continue
        key = line.lower()
        if key in seen and not line.startswith("#"):
            continue
        seen.add(key)
        out.append(line)
    return out


def extract_main_text(html: str, url: str = "", max_chars: int = 6000) -> ExtractedPage:
    """从 HTML 中抽取标题与正文。"""
    page = ExtractedPage(url=url)
    if not html or not html.strip():
        page.error = "empty html"
        return page
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001
        soup = BeautifulSoup(html, "html.parser")

    # 标题
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = str(og["content"]).strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = _visible_text(h1)
    page.title = re.sub(r"\s+", " ", title)[:200]

    desc = soup.find("meta", attrs={"name": "description"})
    if desc and desc.get("content"):
        page.meta["description"] = str(desc["content"]).strip()[:500]
    pub = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find(
        "time"
    )
    if pub:
        page.meta["published"] = str(pub.get("content") or pub.get("datetime") or "").strip()[:40]

    _strip_noise(soup)
    container = _pick_container(soup)
    lines = _clean_lines(_tag_to_lines(container))
    text = _MULTI_NL.sub("\n\n", "\n".join(lines)).strip()

    if len(text) < 200:
        # 主容器定位失败，退回全文
        fallback = _clean_lines(_tag_to_lines(soup.body or soup))
        alt = _MULTI_NL.sub("\n\n", "\n".join(fallback)).strip()
        if len(alt) > len(text):
            text = alt

    if max_chars and len(text) > max_chars:
        cut = text[:max_chars]
        for sep in ("\n\n", "。", ". ", "；", "; "):
            idx = cut.rfind(sep)
            if idx > max_chars * 0.6:
                cut = cut[: idx + len(sep)]
                break
        text = cut.rstrip() + " …[已截断]"

    page.text = text
    page.char_count = len(text)
    page.ok = page.char_count >= 120
    if not page.ok and not page.error:
        page.error = f"正文过短({page.char_count} 字符)，可能是 JS 渲染页或反爬页"
    return page


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
class WebExtractor:
    """带超时、大小上限、UA 伪装、相对链接修正的网页抓取器。"""

    def __init__(self, timeout: int = 15, max_chars: int = 6000, user_agent: str = DEFAULT_UA,
                 max_bytes: int = _MAX_DOWNLOAD_BYTES):
        self.timeout = timeout
        self.max_chars = max_chars
        self.user_agent = user_agent
        self.max_bytes = max_bytes
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "WebExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def fetch(self, url: str) -> ExtractedPage:
        page = ExtractedPage(url=url)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            page.error = f"不支持的协议: {parsed.scheme or '(空)'}"
            return page
        host = (parsed.netloc or "").lower()
        with _HOST_LOCK:
            if host in _HOST_BLOCKED:
                page.error = f"主机 {host} 已熔断（此前连续超时）"
                return page
        try:
            with self.client.stream("GET", url) as resp:
                page.status_code = resp.status_code
                if resp.status_code >= 400:
                    page.error = f"HTTP {resp.status_code}"
                    return page
                ctype = resp.headers.get("content-type", "")
                if ctype and "html" not in ctype and "text" not in ctype:
                    page.error = f"非 HTML 内容: {ctype}"
                    return page
                chunks, total = [], 0
                for chunk in resp.iter_text():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= self.max_bytes:
                        break
                html = "".join(chunks)
        except httpx.TimeoutException:
            self._note_host_failure(host, "超时")
            page.error = "超时"
            return page
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                self._note_host_failure(host, type(exc).__name__)
            page.error = f"{type(exc).__name__}: {exc}"[:200]
            return page

        with _HOST_LOCK:
            _HOST_FAILS.pop(host, None)
        result = extract_main_text(html, url, self.max_chars)
        result.status_code = page.status_code
        if not result.title:
            result.title = _guess_title_from_url(url)
        return result

    @staticmethod
    def _note_host_failure(host: str, reason: str) -> None:
        if not host:
            return
        with _HOST_LOCK:
            _HOST_FAILS[host] = _HOST_FAILS.get(host, 0) + 1
            if _HOST_FAILS[host] >= _HOST_FAIL_THRESHOLD and host not in _HOST_BLOCKED:
                _HOST_BLOCKED.add(host)
                log.warning(f"主机 {host} 连续 {_HOST_FAILS[host]} 次{reason}，本次运行内不再抓取")


def _guess_title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/").split("/")[-1]
    return re.sub(r"[-_]+", " ", re.sub(r"\.\w+$", "", path)).strip() or url


def absolutize(base: str, href: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:  # noqa: BLE001
        return href
