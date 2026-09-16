"""追问对话图：复用调研产出的证据与报告做多轮问答。

与主图的关系：主图产出（证据 + 报告）通过快照/直接注入成为本图的只读上下文，
对话历史用 `add_messages` 累积，独立 thread_id，互不污染调研状态。
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agents.writer import build_evidence_block
from src.config import Settings, get_settings
from src.llm import LLMClient, build_llm
from src.prompts import FOLLOWUP_SYSTEM, FOLLOWUP_USER, LANG_RULE
from src.schemas import Document, FollowupAnswer
from src.state import merge_stats
from src.utils.logger import get_logger

log = get_logger("chat")

try:  # langgraph 版本兼容
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver  # type: ignore


class ChatState(TypedDict, total=False):
    """对话图状态：messages 累积，其余为调研上下文（只读）。"""

    messages: Annotated[list, add_messages]
    topic: str
    evidence: list
    report_markdown: str
    stats: Annotated[dict, merge_stats]
    logs: list


def _last_human(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict):
            role, content = m.get("role") or m.get("type"), m.get("content", "")
        else:
            role, content = getattr(m, "type", None), getattr(m, "content", "")
        if role in ("human", "user") and content:
            return str(content)
    return ""


class FollowupAgent:
    """追问回答节点：只基于已有证据作答，答不了就明说。"""

    name = "followup"

    def __init__(self, chat_model: Any, settings: Optional[Settings] = None):
        self.chat_model = chat_model
        self.settings = settings or get_settings()

    def __call__(self, state: dict) -> dict:
        question = _last_human(state.get("messages") or [])
        topic = state.get("topic", "")
        evidence = [d for d in (state.get("evidence") or []) if getattr(d, "ref", 0)]
        if evidence and isinstance(evidence[0], dict):
            evidence = [Document.model_validate(d) for d in evidence]
        report = (state.get("report_markdown") or "")[:3000]

        system = FOLLOWUP_SYSTEM.format(lang_rule=LANG_RULE)
        user = FOLLOWUP_USER.format(
            topic=topic,
            report_excerpt=report or "（无）",
            evidence_block=build_evidence_block(evidence, per_doc_limit=700, total_limit=5000),
            question=question,
        )

        client = LLMClient(self.chat_model, self.settings)
        before = client.stats.model_dump()
        try:
            answer = client.structured(FollowupAnswer, system, user, task="followup")
        except Exception as exc:  # noqa: BLE001
            log.error(f"追问回答失败: {exc}")
            return {
                "messages": [{"role": "assistant", "content": f"（回答生成失败：{exc}）"}],
                "logs": [f"[chat] 失败: {exc}"],
            }

        text = answer.answer.strip()
        if answer.refs:
            text += f"\n\n（依据：{', '.join(f'[{r}]' for r in answer.refs)}）"
        if not answer.grounded:
            text += "\n\n⚠️ 当前调研证据未覆盖该问题，以上回答不可作为依据。"
        log.info(f"追问完成：grounded={answer.grounded}")
        return {
            "messages": [{"role": "assistant", "content": text}],
            "stats": client.usage_delta(before),
            "logs": [f"[chat] grounded={answer.grounded}"],
        }


def build_chat_graph(chat_model: Optional[Any] = None, settings: Optional[Settings] = None,
                     checkpointer: Any = "auto"):
    settings = settings or get_settings()
    chat_model = chat_model if chat_model is not None else build_llm(settings)
    checkpointer = MemorySaver() if checkpointer == "auto" else checkpointer

    g = StateGraph(ChatState)
    g.add_node("followup", FollowupAgent(chat_model, settings))
    g.add_edge(START, "followup")
    g.add_edge("followup", END)
    return g.compile(checkpointer=checkpointer)
