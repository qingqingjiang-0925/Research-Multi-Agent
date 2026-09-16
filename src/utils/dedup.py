"""去重工具。

两层去重：
1. **URL 精确去重**：同一个页面被多个 query 命中时只保留一份（取质量分更高的）。
2. **正文近似去重**：不同 URL 但内容高度雷同（转载、镜像站、聚合页），
   用字符 n-gram shingle 的 Jaccard 相似度判定。

选 Jaccard + shingle 而不是 embedding 相似度，是因为它零依赖、O(1) 判定、
可解释（能说出"相似度 0.87 所以判重"），对调研场景足够。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff]+")


def normalize(text: str) -> str:
    """小写、去标点、压空白，用于相似度比较。"""
    if not text:
        return ""
    text = _NON_WORD.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def shingles(text: str, k: int = 5) -> frozenset[str]:
    """字符级 k-gram 集合。中文按字、英文按词都适用。"""
    norm = normalize(text)
    if not norm:
        return frozenset()
    if len(norm) <= k:
        return frozenset({norm})
    return frozenset(norm[i : i + k] for i in range(len(norm) - k + 1))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / (len(sa) + len(sb) - inter)


class DedupIndex:
    """增量式近似去重索引。"""

    def __init__(self, threshold: float = 0.72, k: int = 5, min_len: int = 120):
        self.threshold = threshold
        self.k = k
        self.min_len = min_len
        self._items: list[tuple[str, frozenset[str]]] = []

    def __len__(self) -> int:
        return len(self._items)

    def similarity(self, text: str) -> tuple[float, str]:
        """返回与已有条目的最大相似度及其 key。"""
        sh = shingles(text, self.k)
        best, best_key = 0.0, ""
        for key, existing in self._items:
            sim = jaccard(sh, existing)
            if sim > best:
                best, best_key = sim, key
        return best, best_key

    def is_duplicate(self, text: str) -> bool:
        if len(normalize(text)) < self.min_len:
            # 太短的文本（导航页、报错页）不参与近似判重，交给质量过滤处理
            return False
        return self.similarity(text)[0] >= self.threshold

    def add(self, key: str, text: str) -> bool:
        """加入索引。返回 False 表示被判为重复、未加入。"""
        if self.is_duplicate(text):
            return False
        self._items.append((key, shingles(text, self.k)))
        return True


def dedupe_by_key(items, key_fn, score_fn=None):
    """按 key 精确去重，重复时保留 score 更高的那个。保持首次出现顺序。"""
    index: dict = {}
    order: list = []
    for item in items:
        k = key_fn(item)
        if k in index:
            if score_fn is not None and score_fn(item) > score_fn(index[k]):
                index[k] = item
            continue
        index[k] = item
        order.append(k)
    return [index[k] for k in order]
