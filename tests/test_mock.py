"""Mock 层测试：离线 LLM 与离线搜索的确定性。"""

from __future__ import annotations

from src.llm import LLMClient
from src.mock import MockChatModel, mock_search_results
from src.schemas import (
    ExecutiveSummary,
    QueryRewrite,
    ResearchPlan,
    ReviewReport,
    SectionDraft,
)


class TestMockSearch:
    def test_returns_deterministic_results(self):
        a = [(r.url, r.title) for r in mock_search_results("多智能体框架", 5)]
        b = [(r.url, r.title) for r in mock_search_results("多智能体框架", 5)]
        assert a == b
        assert len(a) > 0
        result = mock_search_results("多智能体框架", 5)[0]
        assert result.url.startswith("http")
        assert result.raw_content  # 自带正文，供 writer 引用

    def test_max_results_respected(self):
        assert len(mock_search_results("anything", 3)) == 3

    def test_arbitrary_query_still_finds_something(self):
        # 任意子问题也必须有可检出的证据，保证离线演示不通
        assert len(mock_search_results("完全不存在的主题xyzzy", 5)) >= 2


class TestMockStructured:
    def test_plan(self):
        model = MockChatModel()
        client = LLMClient(model)
        plan = client.structured(
            ResearchPlan,
            "sys",
            "【调研主题】\n多智能体调研系统\n\n其余忽略。",
            task="plan",
        )
        assert isinstance(plan, ResearchPlan)
        assert len(plan.subtasks) >= 3
        assert plan.subtasks[0].id == "T1"
        assert plan.outline[0].id == "S1"
        assert plan.subtasks[0].search_queries

    def test_plan_asks_clarification_on_first_call(self):
        model = MockChatModel(clarify_first=True)
        client = LLMClient(model)
        plan = client.structured(ResearchPlan, "sys", "【调研主题】\nAI 落地", task="plan")
        assert plan.needs_clarification is True
        assert plan.clarification_question

    def test_section_draft_cites_available_refs(self):
        model = MockChatModel()
        client = LLMClient(model)
        user = (
            "【调研主题】\n多智能体\n\n"
            "【本章标题】\n背景\n\n"
            "【证据列表】\n[3] 《A》\n[7] 《B》\n\n【本章写作要求】\n略"
        )
        draft = client.structured(SectionDraft, "sys", user, task="section")
        assert draft.section_id == "S?"  # 真实 section_id 由 WriterAgent 后处理替换
        assert draft.used_refs and draft.content
        assert all(r in (3, 7) for r in draft.used_refs)

    def test_review_passes_by_default(self):
        model = MockChatModel()
        client = LLMClient(model)
        user = "【调研主题】\n多智能体\n\n【证据编号列表】\n[1] [2]\n\n【证据全文】\n[1] 《A》\n\n### S1 标题\n正文[1]"
        report = client.structured(ReviewReport, "sys", user, task="review")
        assert report.passed and report.score >= 0.75
        assert report.fabricated_citations == []

    def test_review_fails_when_configured(self):
        model = MockChatModel(fail_first_review=True)
        client = LLMClient(model)
        user = "【调研主题】\n多智能体\n\n【证据编号列表】\n[1]\n\n【证据全文】\n[1] 《A》\n\n### S1 标题\n正文[1]"
        report = client.structured(ReviewReport, "sys", user, task="review")
        assert not report.passed
        assert report.dirty_sections

    def test_query_rewrite(self):
        model = MockChatModel()
        client = LLMClient(model)
        out = client.structured(QueryRewrite, "sys", "【子问题】\n落地效果", task="rewrite")
        assert len(out.queries) == 3

    def test_followup_grounded_when_refs_exist(self):
        model = MockChatModel()
        client = LLMClient(model)
        user = "【用户追问】\n有什么风险？\n\n【证据列表】\n[1] 《A》\n[2] 《B》"
        out = client.structured(
            __import__("src.schemas", fromlist=["FollowupAnswer"]).FollowupAnswer,
            "sys", user, task="followup",
        )
        assert out.grounded and out.refs
        assert out.answer and not out.answer.startswith("[mock]")

    def test_plain_text_invoke(self):
        model = MockChatModel()
        client = LLMClient(model)
        text = client.text("sys", "hello", task="x")
        assert text.startswith("[mock]")
        assert client.stats.llm_calls == 1