"""报告渲染与落盘测试。"""

from __future__ import annotations

from src.schemas import (
    Document,
    ExecutiveSummary,
    ReviewReport,
    Section,
    SectionDraft,
)
from src.state import initial_state
from src.utils.report import render_report, save_report, slugify


def _rich_state() -> dict:
    st = initial_state("多智能体调研系统", "", "zh")
    st["status"] = "completed"
    st["search_rounds"] = 2
    st["revision_rounds"] = 1
    st["evidence"] = [
        Document(id="d1", url="https://a.example/1", title="来源一", score=0.9, ref=1,
                 content="正文一" * 60),
        Document(id="d2", url="https://a.example/2", title="来源二", score=0.8, ref=2,
                 content="正文二" * 60, published="2025-01-01"),
    ]
    st["sections"] = [Section(id="S1", title="背景", guidance="", order=1, status="final")]
    st["drafts"] = [
        SectionDraft(section_id="S1", content="这是正文，引用来源 [1][2]。", used_refs=[1, 2],
                     confidence=0.8, status="final"),
    ]
    st["summary"] = ExecutiveSummary(
        abstract="本报告调研了多智能体系统。",
        conclusion="建议先用小规模试点。",
        key_findings=["多智能体不是银弹", "需要校验循环"],
    )
    st["review"] = ReviewReport(
        passed=True, score=0.85,
        claims=[], gaps=[], conflicts=[], dirty_sections=[],
        summary="校验通过。",
    )
    st["stats"] = {"llm_calls": 12, "search_calls": 7, "scrape_calls": 3,
                   "prompt_chars": 20000, "completion_chars": 6000}
    return st


def test_slugify():
    assert slugify("多智能体 调研 / 系统 2025!") == "多智能体-调研-系统-2025"
    assert slugify("") == "report"


def test_render_report_contains_core_sections():
    md = render_report(_rich_state())
    assert "# 多智能体调研系统 调研报告" in md
    assert "## 摘要" in md
    assert "## 背景" in md
    assert "## 参考文献" in md
    assert "1. 《来源一》" in md
    assert "## 附录：校验与统计" in md
    assert "✅ 通过" in md
    assert "LLM 调用：12 次" in md


def test_render_report_no_evidence_no_crash():
    st = initial_state("主题", "", "zh")
    st["status"] = "initialized"
    md = render_report(st)
    assert "主题" in md


def test_save_report_writes_file(tmp_path):
    path = save_report("# 标题", "主题 报告", tmp_path)
    assert path.exists()
    assert path.suffix == ".md"
    assert path.read_text(encoding="utf-8").startswith("# 标题")