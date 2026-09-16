"""LangGraph 主图：research 流水线的编排层。

图结构（面试讲解用）：

    START -> planner -> [clarify(interrupt) -> planner]      # 规划 + 人工澄清
                   |-> Send(searcher) x N                    # 每个子任务一个并行分支
         searcher -> collect                                 # 合并证据、分配引用编号
         collect  -> [gap_search -> Send(searcher)]          # 证据不足定向补检索（有界）
                   |-> Send(writer) x M                      # 每章一个并行分支
         writer   -> summarizer -> reviewer                  # 摘要 + 独立校验
         reviewer -> [gap_search]                            # 校验发现缺口 -> 回到检索
                   |-> Send(writer) x dirty                  # 只重写问题章节（有界）
                   |-> finalize -> END                       # 渲染报告、落盘

所有循环都有硬上限（max_search_rounds / max_revision_rounds / max_clarify_rounds），
保证图必然终止；并行分支通过 Send 扇出，状态合并靠 state.py 里的 reducer。
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from src.agents.planner import PlannerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.searcher import SearcherAgent
from src.agents.writer import SummarizeAgent, WriterAgent
from src.config import Settings, get_settings
from src.llm import LLMClient, build_llm
from src.prompts import LANG_RULE, QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER
from src.schemas import QueryRewrite, Section, SubTask
from src.state import ResearchState
from src.utils.logger import get_logger
from src.utils.report import render_report, save_report

log = get_logger("graph")

try:  # langgraph 版本兼容
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver  # type: ignore


def _as_subtask(obj: Any) -> SubTask:
    return obj if isinstance(obj, SubTask) else SubTask.model_validate(obj)


def _as_section(obj: Any) -> Section:
    return obj if isinstance(obj, Section) else Section.model_validate(obj)


class ResearchGraphBuilder:
    """组装 research 图。节点 = Agent 实例（可注入 mock），路由 = 纯函数。"""

    def __init__(self, chat_model: Optional[Any] = None, settings: Optional[Settings] = None,
                 checkpointer: Any = "auto"):
        self.settings = settings or get_settings()
        self.chat_model = chat_model if chat_model is not None else build_llm(self.settings)
        self.checkpointer = MemorySaver() if checkpointer == "auto" else checkpointer

        self.planner = PlannerAgent(self.chat_model, self.settings)
        self.searcher = SearcherAgent(self.chat_model, self.settings)
        self.writer = WriterAgent(self.chat_model, self.settings)
        self.summarizer = SummarizeAgent(self.chat_model, self.settings)
        self.reviewer = ReviewerAgent(self.chat_model, self.settings)

    # ------------------------------------------------------------------ #
    def build(self):
        g = StateGraph(ResearchState)
        g.add_node("planner", self.planner)
        g.add_node("clarify", self._clarify)
        g.add_node("searcher", self.searcher)
        g.add_node("collect", self._collect)
        g.add_node("gap_search", self._gap_search)
        g.add_node("writer", self.writer)
        g.add_node("summarizer", self.summarizer)
        g.add_node("reviewer", self.reviewer)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "planner")
        g.add_conditional_edges("planner", self._route_after_plan)
        g.add_conditional_edges("clarify", self._route_after_clarify)
        g.add_edge("searcher", "collect")
        g.add_conditional_edges("collect", self._route_after_collect)
        g.add_conditional_edges("gap_search", self._fan_out_search)
        g.add_edge("writer", "summarizer")
        g.add_edge("summarizer", "reviewer")
        g.add_conditional_edges("reviewer", self._route_after_review)
        g.add_edge("finalize", END)
        return g.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ #
    # 节点：人工澄清（human-in-the-loop）
    # ------------------------------------------------------------------ #
    def _clarify(self, state: dict) -> dict:
        plan = state.get("plan")
        question = (plan.clarification_question if plan else "") or "请补充调研的范围 / 时间范围 / 用途。"
        answer = interrupt({"question": question, "topic": state.get("topic", "")})
        answer = str(answer or "").strip()
        rounds = int(state.get("clarify_rounds") or 0) + 1
        if answer:
            log.info(f"用户澄清：{answer}")
            return {
                "clarification": answer,
                "clarify_rounds": rounds,
                "logs": [f"[clarify] 用户回答：{answer}"],
            }
        log.info("用户跳过澄清，按默认假设继续")
        return {"clarify_rounds": rounds, "logs": ["[clarify] 用户跳过，按默认假设继续"]}

    # ------------------------------------------------------------------ #
    # 节点：证据合并与引用编号分配
    # ------------------------------------------------------------------ #
    def _collect(self, state: dict) -> dict:
        evidence = list(state.get("evidence") or [])
        subtasks = [_as_subtask(st) for st in (state.get("subtasks") or [])]
        sections = [_as_section(s) for s in (state.get("sections") or [])]
        outcomes = state.get("search_outcomes") or []
        search_rounds = int(state.get("search_rounds") or 0) + 1

        # 1) 总量控制：已编号的必须保留（可能已被引用），新证据按质量分截断
        cap = self.settings.max_evidence_total
        refd = [d for d in evidence if d.ref]
        unrefd = sorted([d for d in evidence if not d.ref], key=lambda d: d.score, reverse=True)
        keep = refd + unrefd[: max(0, cap - len(refd))]

        # 2) 分配引用编号：按子任务顺序 + 质量分，保证可复现
        next_ref = max([d.ref for d in keep] + [0]) + 1
        order_key = {st.id: i for i, st in enumerate(subtasks)}
        ordered = sorted(keep, key=lambda d: (order_key.get(d.subtask_id, 99), -d.score))
        final_docs = []
        for d in ordered:
            if not d.ref:
                d = d.model_copy(update={"ref": next_ref})
                next_ref += 1
            final_docs.append(d)

        # 3) 复核子任务状态（以合并后的证据为准，幂等）
        counts: dict[str, int] = {}
        for d in final_docs:
            counts[d.subtask_id] = counts.get(d.subtask_id, 0) + 1
        new_subtasks = []
        for st in subtasks:
            n = counts.get(st.id, 0)
            status = "done" if n >= self.settings.min_evidence_per_subtask else "gap"
            new_subtasks.append(st.model_copy(update={"status": status}))

        # 4) 本轮被重新检索的子任务 -> 对应章节标记 dirty（触发定向重写）
        wave_ids = {o.get("subtask_id") for o in outcomes if int(o.get("rounds") or 0) == search_rounds}
        new_sections = []
        for sec in sections:
            if sec.status in ("draft", "final") and (set(sec.subtask_ids) & wave_ids):
                sec = sec.model_copy(update={"status": "dirty"})
            new_sections.append(sec)

        gaps = [st for st in new_subtasks if st.status == "gap"]
        log.info(
            f"第 {search_rounds} 轮检索合并完成：{len(final_docs)} 条证据"
            f"（编号至 {next_ref - 1}），{len(gaps)} 个子任务证据不足"
        )
        return {
            "evidence": final_docs,
            "subtasks": new_subtasks,
            "sections": new_sections,
            "search_rounds": search_rounds,
            "status": "searched",
            "logs": [f"[collect] 第 {search_rounds} 轮：{len(final_docs)} 条证据，{len(gaps)} 个子任务证据不足"],
        }

    # ------------------------------------------------------------------ #
    # 节点：把校验缺口翻译成"待补检索的子任务"
    # ------------------------------------------------------------------ #
    def _gap_search(self, state: dict) -> dict:
        review = state.get("review")
        subtasks = [_as_subtask(st) for st in (state.get("subtasks") or [])]
        sections = [_as_section(s) for s in (state.get("sections") or [])]
        gaps = (review.gaps if review else []) or []

        gap_map: dict[str, Any] = {}
        for g in gaps:
            if g.subtask_id:
                gap_map[g.subtask_id] = g
            elif g.section_id:
                for sec in sections:
                    if sec.id == g.section_id:
                        for sid in sec.subtask_ids:
                            gap_map.setdefault(sid, g)
        for st in subtasks:
            if st.status == "gap":
                gap_map.setdefault(st.id, None)
        if not gap_map:  # 缺口无法定位到子任务时，兜底重查所有未完成子任务
            gap_map = {st.id: None for st in subtasks if st.status != "done"} or {st.id: None for st in subtasks}
            log.warning("校验缺口无法映射到子任务，退化为全量补检索")

        updated: list[SubTask] = []
        targets: list[str] = []
        for st in subtasks:
            if st.id not in gap_map:
                updated.append(st)
                continue
            g = gap_map[st.id]
            queries = list(g.suggested_queries[:3]) if g and g.suggested_queries else list(st.search_queries)
            updated.append(
                st.model_copy(
                    update={
                        "status": "pending",
                        "gap_hint": (g.description if g else st.gap_hint) or st.gap_hint,
                        "search_queries": queries or [st.question],
                    }
                )
            )
            targets.append(st.id)

        # 对目标子任务用 LLM 换一批关键词：不换表述，补检索只是重复上一轮的搜索，
        # 全局去重后不会带来新证据——这是反思闭环能否真正收敛的关键。
        stats: dict = {}
        if targets and self.chat_model is not None:
            for i, st in enumerate(updated):
                if st.id not in targets:
                    continue
                delta, fresh = self._rewrite_gap_queries(st)
                if fresh:
                    updated[i] = st.model_copy(update={"search_queries": fresh})
                for k, v in delta.items():
                    stats[k] = stats.get(k, 0) + v

        log.info(f"补检索准备完成：{len(gap_map)} 个子任务待重查")
        result: dict = {
            "subtasks": updated,
            "status": "gap-searching",
            "logs": [f"[gap_search] {len(gap_map)} 个子任务待补检索"],
        }
        if stats:
            result["stats"] = stats
        return result

    def _rewrite_gap_queries(self, st: SubTask) -> tuple[dict, list[str]]:
        """为证据不足的子任务改写关键词。返回 (用量增量, 新关键词)。"""
        client = LLMClient(self.chat_model, self.settings)
        before = client.stats.model_dump()
        system = QUERY_REWRITE_SYSTEM.format(n=3, lang_rule=LANG_RULE)
        user = QUERY_REWRITE_USER.format(
            question=st.question,
            used_queries="\n".join(f"- {q}" for q in st.search_queries) or "（无）",
            found_titles="（无）",
            gap_hint=st.gap_hint or "上一轮证据与其他子任务重复或数量不足，需要全新视角的关键词。",
        )
        try:
            rw = client.structured(QueryRewrite, system, user, task=f"gap-rewrite:{st.id}")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[{st.id}] 补检索关键词改写失败: {exc}")
            return client.usage_delta(before), []
        fresh = [q.strip() for q in (rw.queries or []) if q.strip() and q.strip() not in st.search_queries]
        return client.usage_delta(before), fresh[:3]

    # ------------------------------------------------------------------ #
    # 节点：终态渲染与落盘
    # ------------------------------------------------------------------ #
    def _finalize(self, state: dict) -> dict:
        markdown = render_report(state)
        report_path = ""
        if self.settings.save_report:
            try:
                report_path = str(save_report(markdown, state.get("topic", ""), self.settings.report_dir))
            except Exception as exc:  # noqa: BLE001
                log.error(f"报告保存失败: {exc}")

        sections = [_as_section(s).model_copy(update={"status": "final"}) for s in (state.get("sections") or [])]
        prev = state.get("stats") or {}
        stats_delta = {
            "search_rounds": int(state.get("search_rounds") or 0) - int(prev.get("search_rounds") or 0),
            "revision_rounds": int(state.get("revision_rounds") or 0) - int(prev.get("revision_rounds") or 0),
        }
        status = "error" if state.get("error") else "completed"
        log.info(f"流程结束：{status}，报告 {report_path or '（未保存）'}")
        return {
            "report_markdown": markdown,
            "report_path": report_path,
            "sections": sections,
            "status": status,
            "stats": stats_delta,
            "logs": [f"[finalize] {status}，报告 {report_path or '（未保存）'}"],
        }

    # ------------------------------------------------------------------ #
    # 路由：planner 之后
    # ------------------------------------------------------------------ #
    def _route_after_plan(self, state: dict):
        if state.get("error"):
            return "finalize"
        plan = state.get("plan")
        subtasks = state.get("subtasks") or []
        if not subtasks:
            return "finalize"
        if (
            plan
            and plan.needs_clarification
            and self.settings.human_in_the_loop
            and int(state.get("clarify_rounds") or 0) < self.settings.max_clarify_rounds
        ):
            return "clarify"
        return self._fan_out_search(state)

    def _route_after_clarify(self, state: dict):
        if state.get("clarification"):
            return "planner"
        return self._fan_out_search(state)

    # ------------------------------------------------------------------ #
    # 路由：collect 之后（补检索 or 进入写作）
    # ------------------------------------------------------------------ #
    def _route_after_collect(self, state: dict):
        gaps = [st for st in (state.get("subtasks") or []) if st.status == "gap"]
        if gaps and int(state.get("search_rounds") or 0) < self.settings.max_search_rounds:
            return "gap_search"
        sends = self._fan_out_write(state)
        if sends:
            return sends
        if state.get("drafts"):
            return "summarizer"  # 没有需要（重）写的章节，直接汇总
        return "finalize"

    # ------------------------------------------------------------------ #
    # 路由：reviewer 之后（补检索 / 重写 dirty 章节 / 收尾）
    # ------------------------------------------------------------------ #
    def _route_after_review(self, state: dict):
        review = state.get("review")
        if review is None or review.passed:
            return "finalize"
        if int(state.get("revision_rounds") or 0) > self.settings.max_revision_rounds:
            log.warning("修订轮次用尽，带已知问题收尾")
            return "finalize"
        if review.gaps and int(state.get("search_rounds") or 0) < self.settings.max_search_rounds:
            return "gap_search"
        sends = self._fan_out_rewrite(state)
        return sends if sends else "finalize"

    # ------------------------------------------------------------------ #
    # 扇出：检索（每个待检索子任务一个分支）
    # ------------------------------------------------------------------ #
    def _fan_out_search(self, state: dict) -> list[Send]:
        pending = [st for st in (state.get("subtasks") or []) if st.status in ("pending", "gap")]
        round_ = int(state.get("search_rounds") or 0) + 1
        topic = state.get("topic", "")
        return [
            Send("searcher", {"subtask": st, "topic": topic, "round": round_}) for st in pending
        ]

    # ------------------------------------------------------------------ #
    # 扇出：写作（每个待写章节一个分支）
    # ------------------------------------------------------------------ #
    def _fan_out_write(self, state: dict) -> list[Send]:
        sections = sorted(
            [_as_section(s) for s in (state.get("sections") or [])], key=lambda s: s.order
        )
        todo = [s for s in sections if s.status in ("pending", "dirty")]
        if not todo:
            return []
        evidence = [d for d in (state.get("evidence") or []) if d.ref]
        review = state.get("review")
        notes = self._review_notes(review) if review else {}
        round_ = int(state.get("revision_rounds") or 0) + 1
        topic = state.get("topic", "")
        return [
            Send(
                "writer",
                {
                    "section": s,
                    "topic": topic,
                    "evidence": evidence,
                    "review_notes": notes.get(s.id, ""),
                    "round": round_,
                },
            )
            for s in todo
        ]

    def _fan_out_rewrite(self, state: dict) -> list[Send]:
        review = state.get("review")
        if review is None:
            return []
        sections = {s.id: s for s in (_as_section(s) for s in (state.get("sections") or []))}
        notes = self._review_notes(review)
        targets = [sections[sid] for sid in review.dirty_sections if sid in sections]
        if not targets:
            targets = [s for s in sections.values() if s.status in ("pending", "dirty")]
        if not targets:
            return []
        evidence = [d for d in (state.get("evidence") or []) if d.ref]
        round_ = int(state.get("revision_rounds") or 0) + 1
        topic = state.get("topic", "")
        return [
            Send(
                "writer",
                {
                    "section": s,
                    "topic": topic,
                    "evidence": evidence,
                    "review_notes": notes.get(s.id, ""),
                    "round": round_,
                },
            )
            for s in targets
        ]

    @staticmethod
    def _review_notes(review) -> dict[str, str]:
        """把校验报告按章节整理成 Writer 能执行的修订指令。"""
        notes: dict[str, list[str]] = {}
        for c in review.claims:
            if c.verdict in ("supported",):
                continue
            sid = c.section_id or "S?"
            notes.setdefault(sid, []).append(
                f"断言「{c.text[:60]}」判定为 {c.verdict}（{c.reason}），请修正或删除。"
            )
        for c in review.conflicts:
            notes.setdefault("*", []).append(f"来源冲突：{c.topic} —— {c.description}，请并列呈现分歧。")
        for g in review.gaps:
            if g.section_id:
                notes.setdefault(g.section_id, []).append(f"证据缺口：{g.description}")
        for s in review.suggestions:
            notes.setdefault("*", []).append(s)
        if review.fabricated_citations:
            notes.setdefault("*", []).append(
                f"以下引用编号不存在，必须删除：{review.fabricated_citations}"
            )
        star = notes.pop("*", [])
        return {sid: "\n".join(items + star) for sid, items in notes.items()}


def build_research_graph(chat_model: Optional[Any] = None, settings: Optional[Settings] = None,
                         checkpointer: Any = "auto"):
    """便捷入口：默认内存 checkpointer（支持 interrupt / 断点）。"""
    return ResearchGraphBuilder(chat_model, settings, checkpointer).build()
