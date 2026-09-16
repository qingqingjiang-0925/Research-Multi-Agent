"""报告渲染与落盘。

渲染是纯函数（state -> markdown），方便单测与复用；
落盘只做一件事：起一个可读的文件名，写 UTF-8。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

log = get_logger("report")


def slugify(text: str, max_len: int = 48) -> str:
    """中文友好的文件名 slug：保留字母数字与 CJK，其余折叠成 '-'。"""
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text).strip("-")
    return (text[:max_len].rstrip("-")) or "report"


def render_report(state: dict) -> str:
    """把最终 state 渲染成完整 Markdown 报告（含引用与附录）。"""
    topic = (state.get("topic") or "调研主题").strip()
    summary = state.get("summary")
    sections = sorted(state.get("sections") or [], key=lambda s: s.order)
    drafts = {d.section_id: d for d in (state.get("drafts") or [])}
    evidence = sorted([d for d in (state.get("evidence") or []) if d.ref], key=lambda d: d.ref)
    review = state.get("review")
    stats = state.get("stats") or {}
    error = (state.get("error") or "").strip()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [f"# {topic} 调研报告", ""]
    lines.append(f"> 生成时间：{now} ｜ 状态：{state.get('status', '')}"
                 f" ｜ 检索轮次：{state.get('search_rounds', 0)}"
                 f" ｜ 修订轮次：{state.get('revision_rounds', 0)}")
    lines.append("")

    if error:
        lines += ["> ⚠️ 本次运行存在错误，报告可能不完整：", f"> {error}", ""]

    # ---- 摘要 ----
    if summary:
        lines += ["## 摘要", "", (summary.abstract or "").strip(), ""]
        if summary.key_findings:
            lines += ["## 关键发现", ""]
            lines += [f"- {kf.strip()}" for kf in summary.key_findings if kf.strip()]
            lines.append("")

    # ---- 正文 ----
    for sec in sections:
        draft = drafts.get(sec.id)
        if not draft or not draft.content:
            continue
        lines += [f"## {sec.title}", "", draft.content.strip(), ""]

    # ---- 结论 ----
    if summary and summary.conclusion:
        lines += ["## 结论与建议", "", summary.conclusion.strip(), ""]

    # ---- 参考文献 ----
    if evidence:
        lines += ["## 参考文献", ""]
        for d in evidence:
            pub = f"，{d.published}" if d.published else ""
            lines.append(f"{d.ref}. 《{d.title or d.url}》{pub}")
            lines.append(f"   {d.url}")
        lines.append("")

    # ---- 附录：校验与统计 ----
    lines += ["## 附录：校验与统计", ""]
    if review:
        verdict = "✅ 通过" if review.passed else "❌ 未通过"
        lines += [f"- 校验结论：{verdict}（score={review.score:.2f}）"]
        lines += [f"- 抽取断言：{len(review.claims)} 条；来源冲突：{len(review.conflicts)} 处；"
                  f"证据缺口：{len(review.gaps)} 个；伪造引用：{len(review.fabricated_citations)} 个"]
        if review.claims:
            supported = sum(1 for c in review.claims if c.verdict == "supported")
            lines.append(f"- 断言判定：supported {supported} / {len(review.claims)}")
        if review.conflicts:
            lines += ["", "**来源冲突**", ""]
            lines += [f"- {c.topic}: {c.description}" for c in review.conflicts]
        if review.gaps:
            lines += ["", "**证据缺口（未补齐部分）**", ""]
            lines += [f"- {g.description}" for g in review.gaps]
        if review.suggestions:
            lines += ["", "**改进建议**", ""]
            lines += [f"- {s}" for s in review.suggestions]
        lines.append("")
    else:
        lines += ["- 本次运行未执行校验环节。", ""]

    approx_tokens = int(stats.get("prompt_chars", 0) + stats.get("completion_chars", 0)) / 2.5
    lines += [
        "**运行统计**", "",
        f"- LLM 调用：{stats.get('llm_calls', 0)} 次（约 {int(approx_tokens)} tokens）",
        f"- 搜索调用：{stats.get('search_calls', 0)} 次；网页抓取：{stats.get('scrape_calls', 0)} 次",
        f"- 证据规模：{len(evidence)} 条；章节：{len(sections)} 章",
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def save_report(markdown: str, topic: str, report_dir: Optional[Path] = None) -> Path:
    """写入 reports/ 目录，文件名 = slug + 时间戳。返回路径。"""
    report_dir = Path(report_dir) if report_dir else Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = report_dir / f"{slugify(topic)}-{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    log.info(f"报告已保存: {path}")
    return path


def save_state_snapshot(state: dict, thread_id: str, report_dir: Optional[Path] = None) -> Path:
    """保存追问对话所需的上下文快照（topic / 证据 / 报告正文）。"""
    import json

    report_dir = Path(report_dir) if report_dir else Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence = [d.model_dump() if hasattr(d, "model_dump") else d for d in (state.get("evidence") or [])]
    payload = {
        "thread_id": thread_id,
        "topic": state.get("topic", ""),
        "report_markdown": state.get("report_markdown", ""),
        "evidence": evidence,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = report_dir / f"{slugify(thread_id, max_len=60)}.snapshot.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_state_snapshot(thread_id: str, report_dir: Optional[Path] = None) -> Optional[dict]:
    import json

    report_dir = Path(report_dir) if report_dir else Path("reports")
    path = report_dir / f"{slugify(thread_id, max_len=60)}.snapshot.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning(f"读取快照失败 {path}: {exc}")
        return None
