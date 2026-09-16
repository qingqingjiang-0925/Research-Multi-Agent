"""Search Agent：对单个子任务执行「检索 -> 过滤 -> 正文提取 -> 去重」。

设计要点（面试要点）：
1. **每个子任务一个并行分支**：图用 Send 把 N 个子任务扇出到 N 个 searcher 实例，
   每个实例独立持有 toolkit（独立去重索引），互不共享可变状态，天然线程安全；
   跨子任务的 URL 去重交给全局状态的 `merge_evidence` reducer。
2. **证据不足时自愈**：低于 `min_evidence_per_subtask` 就让 LLM 改写关键词补一轮，
   而不是把"证据不足"静默传给 Writer。
3. **产出可判定**：返回 SearchOutcome（含 status=done/gap），让路由层能基于
   结构化结果决定是否触发全局补检索，而不是靠解析日志。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.config import Settings, get_settings
from src.llm import LLMClient
from src.prompts import LANG_RULE, QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER
from src.schemas import Document, QueryRewrite, SearchOutcome, SubTask
from src.utils.logger import get_logger

log = get_logger("searcher")


class SearcherAgent:
    """检索单分支节点。payload(Send) -> {evidence, subtasks, search_outcomes, ...}"""

    name = "searcher"

    def __init__(self, chat_model: Optional[Any] = None, settings: Optional[Settings] = None,
                 toolkit_factory: Optional[Callable[[], Any]] = None):
        self.chat_model = chat_model
        self.settings = settings or get_settings()
        self._toolkit_factory = toolkit_factory

    # ------------------------------------------------------------------ #
    def _new_toolkit(self):
        if self._toolkit_factory is not None:
            return self._toolkit_factory()
        from src.tools.search import build_toolkit

        return build_toolkit(self.settings)

    # ------------------------------------------------------------------ #
    def __call__(self, payload: dict) -> dict:
        subtask = payload.get("subtask")
        if isinstance(subtask, dict):
            subtask = SubTask.model_validate(subtask)
        topic = payload.get("topic", "")
        wave = int(payload.get("round") or 1)

        toolkit = self._new_toolkit()
        client = LLMClient(self.chat_model, self.settings) if self.chat_model is not None else None
        before_calls, before_scrapes = toolkit.calls, toolkit.scrapes
        before_llm = client.stats.model_dump() if client else {}

        queries = [q.strip() for q in (subtask.search_queries or []) if q.strip()]
        queries = queries[: self.settings.max_queries_per_round] or [subtask.question]

        try:
            docs = toolkit.collect(queries, subtask.id, subtask.question, scrape=True)
            if len(docs) < self.settings.min_evidence_per_subtask and client is not None:
                docs = docs + self._rewrite_and_retry(subtask, toolkit, client, queries, docs)
        except Exception as exc:  # noqa: BLE001
            log.error(f"[{subtask.id}] 检索异常: {exc}")
            docs = []

        enough = len(docs) >= self.settings.min_evidence_per_subtask
        status = "done" if enough else "gap"
        subtask = subtask.model_copy(update={"status": status, "search_queries": queries})
        outcome = SearchOutcome(
            subtask_id=subtask.id,
            documents=docs,
            queries_used=queries,
            rounds=wave,
            status=status,
            note="" if enough else f"仅 {len(docs)} 条证据，低于阈值 {self.settings.min_evidence_per_subtask}",
        )

        stats = client.usage_delta(before_llm) if client else {}
        stats["search_calls"] = toolkit.calls - before_calls
        stats["scrape_calls"] = toolkit.scrapes - before_scrapes

        log.info(f"[{subtask.id}] 第 {wave} 轮：{len(queries)} 组关键词 -> {len(docs)} 条证据（{status}）")
        return {
            "evidence": docs,
            "subtasks": [subtask],
            "search_outcomes": [outcome.model_dump()],
            "stats": stats,
            "logs": [f"[searcher:{subtask.id}] {len(docs)} 条证据（{status}），关键词: {queries}"],
        }

    # ------------------------------------------------------------------ #
    def _rewrite_and_retry(self, subtask: SubTask, toolkit: Any, client: LLMClient,
                           used_queries: list[str], docs: list[Document]) -> list[Document]:
        """证据不足 -> LLM 改写关键词 -> 定向补检索。返回新增的证据。"""
        system = QUERY_REWRITE_SYSTEM.format(n=3, lang_rule=LANG_RULE)
        user = QUERY_REWRITE_USER.format(
            question=subtask.question,
            used_queries="\n".join(f"- {q}" for q in used_queries) or "（无）",
            found_titles="\n".join(f"- {d.title}" for d in docs[:5]) or "（无）",
            gap_hint=subtask.gap_hint or "证据数量不足，需要更具体或换语言的关键词。",
        )
        try:
            rewrite = client.structured(QueryRewrite, system, user, task=f"rewrite:{subtask.id}")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[{subtask.id}] 查询改写失败: {exc}")
            return []

        fresh = [q.strip() for q in (rewrite.queries or []) if q.strip() and q.strip() not in used_queries]
        if not fresh:
            return []
        log.info(f"[{subtask.id}] 改写关键词: {fresh}")
        return toolkit.collect(fresh, subtask.id, subtask.question, scrape=True)
