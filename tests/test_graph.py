"""端到端图测试：离线 mock 全流程、修订循环、人工澄清、追问对话。"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from src.config import Settings
from src.mock import MockChatModel
from src.state import initial_state


def _settings(**overrides) -> Settings:
    base = dict(
        llm_provider="mock",
        search_backend="mock",
        scrape_enabled=False,
        save_report=False,
        log_level="WARNING",
        human_in_the_loop=False,
    )
    base.update(overrides)
    return Settings(**base)


def _invoke(graph, state, thread_id: str):
    return graph.invoke(state, config={"configurable": {"thread_id": thread_id}})


class TestFullPipeline:
    def test_offline_run_completes(self):
        from src.graph.builder import build_research_graph

        graph = build_research_graph(MockChatModel(), _settings())
        tid = "test-full-1"
        result = _invoke(graph, initial_state("多智能体深度调研系统", "", "zh"), tid)

        assert result["status"] == "completed"
        assert not result["error"]
        # 证据：全部拿到连续引用编号
        evidence = [d for d in result["evidence"] if d.ref]
        assert len(evidence) >= 5
        assert {d.ref for d in evidence} == set(range(1, len(evidence) + 1))
        # 章节与草稿（mock 每章都有产出）
        assert len(result["sections"]) >= 3
        drafts = [d for d in result["drafts"] if d.content]
        assert len(drafts) >= 3
        # 校验通过
        assert result["review"] is not None and result["review"].passed
        # 报告
        assert "参考文献" in result["report_markdown"]
        assert result["report_path"] == ""  # save_report=False

    def test_gap_loop_and_revision_loop_converge(self):
        """fail_first_review 先触发校验发现的 gap 补检索，再触发重写，最终收敛。"""
        from src.graph.builder import build_research_graph

        model = MockChatModel(fail_first_review=True)
        graph = build_research_graph(model, _settings())
        tid = "test-loops-1"
        result = _invoke(graph, initial_state("多智能体框架对比", "", "zh"), tid)

        assert result["status"] == "completed"
        assert result["search_rounds"] >= 2  # 发生过补检索
        assert result["revision_rounds"] >= 1  # 发生过修订
        assert len(result["review_history"]) >= 2
        assert result["review"].passed
        assert result["search_rounds"] <= 3
        assert result["revision_rounds"] <= 2

    def test_loops_are_bounded(self):
        """校验永远失败时，修订次数有硬上限，图必然终止而不是死循环。"""
        import src.mock as mock_mod
        from src.graph.builder import build_research_graph
        from src.schemas import Claim

        orig = mock_mod._GENERATORS["ReviewReport"]

        def _always_fail(model, user):
            report = orig(model, user)
            return report.model_copy(
                update={
                    "passed": False,
                    "score": 0.3,
                    "claims": [
                        Claim(id="C1", section_id="S1", text="审查断言", verdict="unsupported",
                              severity="high", reason="test: always fail")
                    ],
                    "section_verdicts": {"S1": 0.0},
                    "dirty_sections": ["S1"],
                    "gaps": [],
                }
            )

        mock_mod._GENERATORS["ReviewReport"] = _always_fail
        try:
            graph = build_research_graph(MockChatModel(), _settings())
            tid = "test-bounded-1"
            result = _invoke(graph, initial_state("有界性验证", "", "zh"), tid)
        finally:
            mock_mod._GENERATORS["ReviewReport"] = orig

        # 最多 max_revision_rounds(2) 次重写 + 1 次终态校验
        assert result["status"] == "completed"
        assert len(result["review_history"]) <= 3
        assert result["revision_rounds"] <= 3

    def test_section_drafts_cite_valid_refs(self):
        """写出来的章节只能引用合法编号，这是不依赖模型自觉的硬校验。"""
        from src.graph.builder import build_research_graph

        graph = build_research_graph(MockChatModel(), _settings())
        tid = "test-refs-1"
        result = _invoke(graph, initial_state("多智能体", "", "zh"), tid)

        valid = {d.ref for d in result["evidence"] if d.ref}
        for draft in result["drafts"]:
            for used in draft.used_refs:
                assert used in valid
            assert not draft.content.startswith("```")


class TestClarify:
    def test_human_in_the_loop_interrupt_and_resume(self):
        from src.graph.builder import build_research_graph

        model = MockChatModel(clarify_first=True)
        graph = build_research_graph(model, _settings(human_in_the_loop=True))
        tid = "test-clarify-1"

        result = _invoke(graph, initial_state("多智能体调研", "", "zh"), tid)
        assert "__interrupt__" in result
        intr = result["__interrupt__"][0]
        assert intr.value["question"]

        # 用户回答后继续（Command.resume 让 interrupt 返回该回答）
        result = graph.invoke(
            Command(resume="更关注选型对比"),
            config={"configurable": {"thread_id": tid}},
        )
        assert "__interrupt__" not in result
        assert result["clarification"] == "更关注选型对比"
        assert result["clarify_rounds"] == 1
        assert result["status"] == "completed"

    def test_no_hitl_skips_interrupt(self):
        from src.graph.builder import build_research_graph

        model = MockChatModel(clarify_first=True)
        graph = build_research_graph(model, _settings(human_in_the_loop=False))
        tid = "test-no-hitl-1"
        result = _invoke(graph, initial_state("多智能体调研", "", "zh"), tid)
        assert "__interrupt__" not in result
        assert result["status"] == "completed"


class TestChatGraph:
    def _seed(self):
        from src.schemas import Document

        return {
            "topic": "多智能体框架",
            "evidence": [
                Document(id="x1", url="https://a.example/1", title="A", ref=1, score=0.9,
                         content="多智能体系统进入工程化阶段" + "。" * 30),
                Document(id="x2", url="https://a.example/2", title="B", ref=2, score=0.8,
                         content="缺少统一可复现基准" + "。" * 30),
            ],
            "report_markdown": "## 摘要\n这是报告。",
            "messages": [],
        }

    def test_followup_uses_evidence(self):
        from src.graph.chat import build_chat_graph

        graph = build_chat_graph(MockChatModel(), _settings())
        tid = "test-chat-1"
        config = {"configurable": {"thread_id": tid}}
        payload = {"messages": [{"role": "user", "content": "当前主要挑战是什么？"}]}
        payload.update({k: v for k, v in self._seed().items() if k != "messages"})
        result = graph.invoke(payload, config=config)
        last = result["messages"][-1]
        content = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
        assert "基准" in content
        assert "[1]" in content  # 引用编号跟在回答里

    def test_followup_no_evidence_says_grounded_false(self):
        from src.graph.chat import build_chat_graph

        graph = build_chat_graph(MockChatModel(), _settings())
        tid = "test-chat-2"
        config = {"configurable": {"thread_id": tid}}
        payload = {"messages": [{"role": "user", "content": "有什么风险？"}]}
        payload.update({"topic": "主题", "evidence": [], "report_markdown": ""})
        result = graph.invoke(payload, config=config)
        last = result["messages"][-1]
        content = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
        assert "证据中没有覆盖" in content