"""Planner Agent：把模糊的调研需求拆解成可执行的检索计划。

职责边界（面试要点）：只做拆解与规划，不做检索、不写正文。
产出 `ResearchPlan`（子任务 + 大纲），是后续所有并行分支的"任务清单"。
"""

from __future__ import annotations

from typing import Any, Optional

from src.config import Settings, get_settings
from src.llm import LLMClient
from src.prompts import (
    LANG_RULE,
    PLANNER_CLARIFICATION_BLOCK,
    PLANNER_SYSTEM,
    PLANNER_USER,
)
from src.schemas import ResearchPlan
from src.utils.logger import get_logger

log = get_logger("planner")


class PlannerAgent:
    """规划节点。state -> {plan, subtasks, sections, ...}"""

    name = "planner"

    def __init__(self, chat_model: Any, settings: Optional[Settings] = None):
        self.chat_model = chat_model
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    def __call__(self, state: dict) -> dict:
        topic = (state.get("topic") or "").strip()
        user_context = (state.get("user_context") or "").strip()
        clarification = (state.get("clarification") or "").strip()
        clarify_rounds = int(state.get("clarify_rounds") or 0)

        system = PLANNER_SYSTEM.format(
            min_subtasks=self.settings.min_subtasks,
            max_subtasks=self.settings.max_subtasks,
            lang_rule=LANG_RULE,
        )
        block = PLANNER_CLARIFICATION_BLOCK.format(clarification=clarification) if clarification else ""
        user = PLANNER_USER.format(
            topic=topic,
            user_context=user_context or "（无）",
            clarification_block=block,
        )

        client = LLMClient(self.chat_model, self.settings)
        before = client.stats.model_dump()
        try:
            plan = client.structured(ResearchPlan, system, user, task="planner")
        except Exception as exc:  # noqa: BLE001
            log.error(f"Planner 失败: {exc}")
            return {
                "error": f"Planner 失败: {exc}",
                "status": "error",
                "stats": client.usage_delta(before),
                "logs": [f"[planner] 失败: {exc}"],
            }

        plan = self.normalize(plan, topic, clarify_rounds)
        log.info(f"规划完成：{len(plan.subtasks)} 个子任务 / {len(plan.outline)} 章"
                 + ("（需要追问）" if plan.needs_clarification else ""))
        return {
            "plan": plan,
            "subtasks": plan.subtasks,
            "sections": plan.outline,
            "status": "planned",
            "stats": client.usage_delta(before),
            "logs": [f"[planner] 拆解为 {len(plan.subtasks)} 个子任务、{len(plan.outline)} 章大纲"],
        }

    # ------------------------------------------------------------------ #
    def normalize(self, plan: ResearchPlan, topic: str, clarify_rounds: int) -> ResearchPlan:
        """确定性后处理：不信任 LLM 的 id / 数量 / 引用关系，全部重写一遍。"""
        if not (plan.topic or "").strip():
            plan.topic = topic

        subs = plan.subtasks[: self.settings.max_subtasks]
        for i, st in enumerate(subs, 1):
            st.id = f"T{i}"
            queries = [q.strip() for q in (st.search_queries or []) if q and q.strip()]
            st.search_queries = queries[:3] or [st.question]
            st.status = "pending"
            st.gap_hint = ""
        if len(subs) < self.settings.min_subtasks:
            log.warning(f"子任务数量 {len(subs)} 低于下限 {self.settings.min_subtasks}")
        plan.subtasks = subs

        valid_ids = {st.id for st in subs}
        for i, sec in enumerate(plan.outline, 1):
            sec.id = f"S{i}"
            sec.order = i
            sec.subtask_ids = [sid for sid in sec.subtask_ids if sid in valid_ids]
            sec.status = "pending"
        plan.outline = plan.outline[:10]

        # 追问次数用尽后强制不再追问，保证流程有界
        if clarify_rounds >= self.settings.max_clarify_rounds:
            plan.needs_clarification = False
        return plan
