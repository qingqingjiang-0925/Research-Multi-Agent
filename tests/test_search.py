"""搜索工具层测试：质量打分、mock 后端、toolkit 管线。"""

from __future__ import annotations

from src.mock import MOCK_CORPUS
from src.schemas import make_doc_id
from src.tools.search import (
    MockSearchBackend,
    SearchResult,
    SearchToolkit,
    build_toolkit,
    score_result,
)


class TestScoreResult:
    def base(self, url="https://example.com/a", title="多智能体 框架 对比", snippet="x" * 120):
        return SearchResult(title=title, url=url, snippet=snippet)

    def test_trusted_domain_boost(self):
        s = self.base(url="https://arxiv.org/a")
        assert score_result(s, "多智能体") > score_result(self.base(), "多智能体")

    def test_blocked_domain_penalty(self):
        s = self.base(url="https://facebook.com/a")
        assert score_result(s, "多智能体") < score_result(self.base(), "多智能体")

    def test_nonsense_page_penalty(self):
        s = self.base(title="404 not found 登录 验证码")
        assert score_result(s, "多智能体") < 0.25

    def test_returns_in_01(self):
        for q in ("多智能体", "langgraph", "zzzz"):
            v = score_result(self.base(), q)
            assert 0.0 <= v <= 1.0

    def test_query_relevance_counts(self):
        hit = self.base(title="LangGraph 多智能体 框架 对比 评测")
        miss = self.base(title="量子计算 纠错码 综述")
        assert score_result(hit, "多智能体 框架 对比") > score_result(miss, "多智能体 框架 对比")


class TestMockBackend:
    def test_corpus_hits_relevant_query(self):
        backend = MockSearchBackend()
        hits = backend.search("LangGraph", 6)
        urls = [r.url for r in hits]
        assert any("langgraph" in u for u in urls)

    def test_custom_corpus_injection(self):
        backend = MockSearchBackend(corpus=[MOCK_CORPUS[0]])
        out = backend.search("anything", 5)
        assert len(out) == 1
        assert out[0].url == MOCK_CORPUS[0]["url"]


class TestToolkit:
    def _toolkit(self):
        return SearchToolkit(backend=MockSearchBackend(), extractor=None)

    def test_collect_produces_refable_docs(self):
        tk = self._toolkit()
        docs = tk.collect(["多智能体框架对比"], subtask_id="T1", question="多智能体框架对比")
        assert docs
        for d in docs:
            assert d.subtask_id == "T1"
            assert d.best_text
            assert len(make_doc_id(d.url)) == 12

    def test_limit_and_seen_urls(self):
        tk = self._toolkit()
        docs = tk.collect(["多智能体", "LangGraph", "AutoGen"], subtask_id="T1", limit=4)
        assert len(docs) <= 4
        # 第二次检索不应重复返回已见过的 URL
        more = tk.collect(["多智能体", "LangGraph"], subtask_id="T1", limit=4)
        assert all(d.id not in {x.id for x in docs} for d in more)

    def test_queries_deduped_across_rounds_via_seen(self):
        tk = self._toolkit()
        tk.collect(["deep research"], subtask_id="T1")
        second = tk.collect(["deep research"], subtask_id="T1")
        assert second == []


def test_build_toolkit_mock_disables_extractor():
    from src.config import Settings

    settings = Settings(llm_provider="mock", search_backend="mock", scrape_enabled=True)
    tk = build_toolkit(settings)
    assert tk.extractor is None  # mock URL 不可抓取，必须禁用