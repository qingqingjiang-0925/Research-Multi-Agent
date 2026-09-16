"""Writer Agent（含 Summarizer）：只基于证据写作，逐句溯源。

设计要点（面试要点）：
1. **证据先行**：Writer 收到的是已经过打分、去重、编号的证据列表，
   prompt 里硬性禁止外部知识，事实句必须带 [n] 引用；
2. **并行写作**：每章一个分支（Send 扇出），章与章之间无依赖，
   修订时只重写 dirty 章节，不整篇重算；
3. **确定性后处理**：used_refs 与正文中的 [n] 交叉校验，
   编号不在合法集合里的一律剔除——不靠模型自觉。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from src.config import Settings, get_settings
from src.llm import LLMClient
from src.prompts import (
    EVIDENCE_ITEM,
    LANG_RULE,
    SUMMARY_SYSTEM,
    SUMMARY_USER,
    WRITER_REVIEW_BLOCK,
    WRITER_SYSTEM,
    WRITER_USER,
)
from src.schemas import Document, ExecutiveSummary, Section, SectionDraft
from src.utils.logger import get_logger

log = get_logger("writer")

_CITED = re.compile(r"\[(\d+)\]")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|^```\s*$", re.M)


def build_evidence_block(docs: list[Document], per_doc_limit: int = 1400,
                         total_limit: int = 9000) -> str:
    """把证据渲染成 Writer / Reviewer 共用的编号文本块。"""
    if not docs:
        return "（当前没有可用证据）"
    blocks: list[str] = []
    total = 0
    for d in docs:
        content = (d.best_text or "").strip()[:per_doc_limit]
        block = EVIDENCE_ITEM.format(
            ref=d.ref,
            title=d.title or d.url,
            url=d.url,
            published=d.published or "未知",
            content=content or "（无正文）",
        )
        blocks.append(block)
        total += len(block)
        if total >= total_limit:
            break
    return "\n\n".join(blocks)


class WriterAgent:
    """单章写作节点。payload(Send) -> {drafts, sections, ...}"""

    name = "writer"

    def __init__(self, chat_model: Any, settings: Optional[Settings] = None):
        self.chat_model = chat_model
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    def __call__(self, payload: dict) -> dict:
        section = payload.get("section")
        if isinstance(section, dict):
            section = Section.model_validate(section)
        topic = payload.get("topic", "")
        evidence = payload.get("evidence") or []
        if evidence and isinstance(evidence[0], dict):
            evidence = [Document.model_validate(d) for d in evidence]
        review_notes = (payload.get("review_notes") or "").strip()
        wave = int(payload.get("round") or 1)

        docs = self._select_evidence(section, evidence)
        valid_refs = sorted({d.ref for d in docs if d.ref})
        evidence_block = build_evidence_block(docs)

        system = WRITER_SYSTEM.format(lang_rule=LANG_RULE)
        review_block = WRITER_REVIEW_BLOCK.format(notes=review_notes) if review_notes else ""
        user = WRITER_USER.format(
            topic=topic,
            section_title=section.title,
            section_guidance=section.guidance or "（无特别要求）",
            evidence_block=evidence_block,
            review_block=review_block,
        )

        client = LLMClient(self.chat_model, self.settings)
        before = client.stats.model_dump()
        try:
            draft = client.structured(SectionDraft, system, user, task=f"writer:{section.id}")
        except Exception as exc:  # noqa: BLE001
            log.error(f"[{section.id}] 写作失败: {exc}")
            draft = SectionDraft(
                section_id=section.id,
                content=f"（本章生成失败：{exc}）",
                used_refs=[],
                confidence=0.0,
                missing_info=f"生成异常: {exc}",
            )

        draft = self._postprocess(draft, section, valid_refs)
        section = section.model_copy(update={"status": "draft"})
        log.info(f"[{section.id}] 第 {wave} 轮写作完成，引用 {draft.used_refs}，置信度 {draft.confidence}")
        return {
            "drafts": [draft],
            "sections": [section],
            "stats": client.usage_delta(before),
            "logs": [f"[writer:{section.id}] {len(draft.content)} 字，引用 {len(draft.used_refs)} 条"
                     + (f"，按校验意见修订" if review_notes else "")],
        }

    # ------------------------------------------------------------------ #
    def _select_evidence(self, section: Section, evidence: list[Document]) -> list[Document]:
        """选支撑本章的证据：按 subtask_ids 过滤，无映射时退回全量，按质量分排序。"""
        docs = [d for d in evidence if d.ref]
        if section.subtask_ids:
            picked = [d for d in docs if d.subtask_id in section.subtask_ids]
            if picked:
                docs = picked
        docs = sorted(docs, key=lambda d: d.score, reverse=True)
        cap = max(self.settings.max_evidence_per_subtask + 2, 4)
        return docs[:cap]

    def _postprocess(self, draft: SectionDraft, section: Section, valid_refs: list[int]) -> SectionDraft:
        """确定性清洗：去围栏/标题、校验引用编号、夹置信度。"""
        content = _FENCE.sub("", draft.content or "").strip()
        # 去掉模型自作主张加的章节标题（系统会统一加）
        title_pat = re.compile(r"^#{1,6}\s*" + re.escape(section.title.strip()) + r"\s*$", re.M)
        content = title_pat.sub("", content).strip()

        valid = set(valid_refs)
        cited = {int(n) for n in _CITED.findall(content)}
        used = [r for r in draft.used_refs if r in valid]
        for r in sorted(cited & valid):
            if r not in used:
                used.append(r)
        if not used and valid_refs:
            log.warning(f"[{section.id}] 未产生合法引用，标记低置信度")

        return draft.model_copy(
            update={
                "section_id": section.id,
                "content": content,
                "used_refs": sorted(set(used)),
                "confidence": round(min(max(draft.confidence, 0.0), 1.0), 2),
            }
        )


class SummarizeAgent:
    """摘要与结论节点（串行，依赖全部章节完成）。state -> {summary, ...}"""

    name = "summarizer"

    def __init__(self, chat_model: Any, settings: Optional[Settings] = None):
        self.chat_model = chat_model
        self.settings = settings or get_settings()

    def __call__(self, state: dict) -> dict:
        topic = state.get("topic", "")
        sections = sorted(state.get("sections") or [], key=lambda s: s.order)
        drafts = {d.section_id: d for d in (state.get("drafts") or [])}
        review = state.get("review")

        parts = []
        for sec in sections:
            draft = drafts.get(sec.id)
            if draft and draft.content:
                parts.append(f"## {sec.title}\n{draft.content}")
        sections_block = "\n\n".join(parts) or "（暂无章节正文）"
        review_summary = (review.summary if review else "") or "（首轮生成，暂无校验结论）"

        client = LLMClient(self.chat_model, self.settings)
        before = client.stats.model_dump()
        try:
            summary = client.structured(
                ExecutiveSummary,
                SUMMARY_SYSTEM.format(lang_rule=LANG_RULE),
                SUMMARY_USER.format(topic=topic, sections_block=sections_block, review_summary=review_summary),
                task="summary",
            )
        except Exception as exc:  # noqa: BLE001
            log.error(f"摘要生成失败: {exc}")
            summary = ExecutiveSummary(
                abstract="（摘要生成失败）",
                conclusion="（结论生成失败）",
                key_findings=[],
            )

        log.info(f"摘要完成：{len(summary.key_findings)} 条关键发现")
        return {
            "summary": summary,
            "stats": client.usage_delta(before),
            "logs": [f"[summarizer] 摘要 {len(summary.abstract)} 字，关键发现 {len(summary.key_findings)} 条"],
        }
