"""LangGraph 全局状态定义。

设计要点：
- 用 `Annotated` + reducer 让并行分支（Send 扇出）的写入可以安全合并；
- 证据 / 子任务 / 章节都按 id 合并，天然幂等，重跑某一轮不会产生脏数据；
- `stats` 累加，用于 token 成本统计与终止控制。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages

from src.schemas import (
    Document,
    ExecutiveSummary,
    ResearchPlan,
    ReviewReport,
    Section,
    SectionDraft,
    Stats,
    SubTask,
)


def merge_by_id(field: str = "id"):
    """通用 reducer：按 id 合并列表，后写覆盖先写，保持首次出现顺序。"""

    def _merge(left: Optional[list], right: Optional[list]) -> list:
        left = left or []
        right = right or []
        index: dict[Any, Any] = {}
        order: list[Any] = []
        for item in list(left) + list(right):
            key = getattr(item, field, None)
            if key is None:
                order.append(object())
                index[id(order[-1])] = item
                continue
            if key not in index:
                order.append(key)
            index[key] = item
        return [index[k] for k in order]

    return _merge


def merge_evidence(left: Optional[list[Document]], right: Optional[list[Document]]) -> list[Document]:
    """证据合并：同 URL 保留质量分更高、正文更长的那条。"""
    left = left or []
    right = right or []
    index: dict[str, Document] = {}
    order: list[str] = []

    def _rank(d: Document) -> tuple[float, int]:
        return (round(d.score, 4), len(d.best_text))

    for doc in list(left) + list(right):
        if doc.id not in index:
            order.append(doc.id)
            index[doc.id] = doc
            continue
        old = index[doc.id]
        new = doc.model_copy()
        # 保留已分配的参考文献编号
        if old.ref and not new.ref:
            new.ref = old.ref
        if old.subtask_id and not new.subtask_id:
            new.subtask_id = old.subtask_id
        if _rank(new) >= _rank(old):
            index[doc.id] = new
        else:
            keep = old.model_copy()
            if new.scraped:
                keep.content = new.content
                keep.scraped = True
            index[doc.id] = keep
    return [index[k] for k in order]


def merge_stats(left: Optional[dict], right: Optional[dict]) -> dict:
    base = Stats(**(left or {}))
    delta = Stats(**(right or {}))
    return base.merged(delta).model_dump()


def replace(_left: Any, right: Any) -> Any:
    return right


class ResearchState(TypedDict, total=False):
    """整张图共享的状态。"""

    # ---- 输入 ----
    topic: str
    user_context: str
    language: str

    # ---- 规划 ----
    plan: Optional[ResearchPlan]
    subtasks: Annotated[list[SubTask], merge_by_id("id")]
    sections: Annotated[list[Section], merge_by_id("id")]
    clarify_rounds: int
    clarification: str

    # ---- 检索 ----
    evidence: Annotated[list[Document], merge_evidence]
    search_outcomes: Annotated[list[dict], operator.add]

    # ---- 写作 ----
    drafts: Annotated[list[SectionDraft], merge_by_id("section_id")]
    summary: Optional[ExecutiveSummary]

    # ---- 校验 ----
    review: Optional[ReviewReport]
    review_history: Annotated[list[ReviewReport], operator.add]

    # ---- 输出 ----
    report_markdown: str
    report_path: str

    # ---- 控制 ----
    search_rounds: int
    revision_rounds: int
    status: str
    error: str
    stats: Annotated[dict, merge_stats]
    logs: Annotated[list[str], operator.add]

    # ---- 多轮对话 ----
    messages: Annotated[list, add_messages]
    thread_topic: str


def initial_state(topic: str, user_context: str = "", language: str = "zh") -> ResearchState:
    return ResearchState(
        topic=topic.strip(),
        user_context=(user_context or "").strip(),
        language=language,
        plan=None,
        subtasks=[],
        sections=[],
        clarify_rounds=0,
        clarification="",
        evidence=[],
        search_outcomes=[],
        drafts=[],
        summary=None,
        review=None,
        review_history=[],
        report_markdown="",
        report_path="",
        search_rounds=0,
        revision_rounds=0,
        status="initialized",
        error="",
        stats=Stats().model_dump(),
        logs=[],
        messages=[],
    )
