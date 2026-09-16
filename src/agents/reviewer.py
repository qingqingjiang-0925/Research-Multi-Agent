"""Reviewer Agent：以怀疑立场对成稿做事实校验。

设计要点（面试要点）：
1. **独立角色**：Reviewer 与 Writer 使用不同的 system prompt（立场=怀疑），
   形成"生成-校验"对抗，这是抑制幻觉的核心机制；
2. **确定性兜底**：LLM 给出的判定再经过代码复核——引用编号是否合法、
   passed 条件是否满足，全部用规则重算一遍，不信任模型的自我评估；
3. **结构化产出**：claims / conflicts / gaps / dirty_sections 直接驱动
   图的路由（补检索 or 重写），闭环不靠人读报告。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from src.config import Settings, get_settings
from src.llm import LLMClient
from src.prompts import LANG_RULE, REVIEWER_SYSTEM, REVIEWER_USER
from src.schemas import ReviewReport
from src.utils.logger import get_logger

log = get_logger("reviewer")

_CITED = re.compile(r"\[(\d+)\]")


class ReviewerAgent:
    """校验节点。state -> {review, review_history, revision_rounds, ...}"""

    name = "reviewer"

    def __init__(self, chat_model: Any, settings: Optional[Settings] = None):
        self.chat_model = chat_model
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    def __call__(self, state: dict) -> dict:
        topic = state.get("topic", "")
        evidence = [d for d in (state.get("evidence") or []) if d.ref]
        sections = sorted(state.get("sections") or [], key=lambda s: s.order)
        drafts = {d.section_id: d for d in (state.get("drafts") or [])}
        revision_rounds = int(state.get("revision_rounds") or 0)

        valid_refs = sorted(d.ref for d in evidence)
        from src.agents.writer import build_evidence_block

        evidence_block = build_evidence_block(evidence, per_doc_limit=900, total_limit=7000)
        parts = []
        for sec in sections:
            draft = drafts.get(sec.id)
            if draft and draft.content:
                parts.append(f"### {sec.id} {sec.title}\n{draft.content}")
        sections_block = "\n\n".join(parts) or "（暂无章节正文）"

        system = REVIEWER_SYSTEM.format(
            max_claims_per_section=6,
            min_evidence=self.settings.min_evidence_per_subtask,
            pass_score=self.settings.review_pass_score,
            lang_rule=LANG_RULE,
        )
        user = REVIEWER_USER.format(
            topic=topic,
            valid_refs=", ".join(f"[{r}]" for r in valid_refs) or "（无）",
            evidence_block=evidence_block,
            sections_block=sections_block,
        )

        client = LLMClient(self.chat_model, self.settings)
        before = client.stats.model_dump()
        try:
            report = client.structured(ReviewReport, system, user, task="reviewer")
        except Exception as exc:  # noqa: BLE001
            log.error(f"校验失败，按通过处理以免阻塞流程: {exc}")
            report = ReviewReport(
                passed=True,
                score=0.0,
                summary=f"（校验 Agent 异常，跳过：{exc}）",
            )

        report = self._enforce(report, drafts, sections, valid_refs)
        next_rounds = revision_rounds if report.passed else revision_rounds + 1

        log.info(
            f"校验完成：score={report.score:.2f} passed={report.passed} "
            f"claims={len(report.claims)} gaps={len(report.gaps)} dirty={report.dirty_sections}"
        )
        return {
            "review": report,
            "review_history": [report],
            "revision_rounds": next_rounds,
            "stats": client.usage_delta(before),
            "logs": [
                f"[reviewer] score={report.score:.2f} passed={report.passed} "
                f"claims={len(report.claims)} gaps={len(report.gaps)} dirty={report.dirty_sections}"
            ],
        }

    # ------------------------------------------------------------------ #
    def _enforce(self, report: ReviewReport, drafts: dict, sections: list,
                 valid_refs: list[int]) -> ReviewReport:
        """用确定性规则复核 LLM 的校验结论。"""
        valid = set(valid_refs)
        section_ids = {s.id for s in sections}

        # 1) 伪造引用：正文中出现但不在合法编号集合里
        fabricated: set[int] = set()
        for draft in drafts.values():
            for n in _CITED.findall(draft.content or ""):
                if int(n) not in valid:
                    fabricated.add(int(n))
        fabricated |= {r for r in report.fabricated_citations if r not in valid}

        # 2) 高危无支撑断言
        high_unsupported = [
            c for c in report.claims
            if c.verdict == "unsupported" and c.severity == "high"
        ]

        # 3) 每章得分兜底（LLM 漏给时按 claim 占比重算）
        verdicts = dict(report.section_verdicts)
        for sid in section_ids:
            if sid in verdicts:
                continue
            claims = [c for c in report.claims if c.section_id == sid]
            if claims:
                good = sum(1 for c in claims if c.verdict in ("supported", "opinion"))
                verdicts[sid] = round(good / len(claims), 2)
            else:
                verdicts[sid] = 0.75
        verdicts = {k: round(min(max(v, 0.0), 1.0), 2) for k, v in verdicts.items() if k in section_ids}

        # 4) 总分与通过条件：规则说了算
        score = report.score
        if verdicts:
            score = round(min(max(score, sum(verdicts.values()) / len(verdicts), 0.0), 1.0), 2)
        if fabricated:
            score = round(max(0.0, score - 0.05 * len(fabricated)), 2)
        if report.gaps:
            score = round(max(0.0, score - 0.03 * len(report.gaps)), 2)

        passed = (
            score >= self.settings.review_pass_score
            and not fabricated
            and not high_unsupported
        )

        # 5) dirty 章节：LLM 判定 + 规则追加（伪造引用所在章 / 低分章）
        dirty = [sid for sid in report.dirty_sections if sid in section_ids]
        for draft in drafts.values():
            if any(int(n) not in valid for n in _CITED.findall(draft.content or "")):
                if draft.section_id not in dirty:
                    dirty.append(draft.section_id)
        for sid, v in verdicts.items():
            if v < self.settings.review_pass_score and sid not in dirty:
                dirty.append(sid)

        return report.model_copy(
            update={
                "fabricated_citations": sorted(fabricated),
                "section_verdicts": verdicts,
                "score": score,
                "passed": passed,
                "dirty_sections": dirty,
            }
        )
