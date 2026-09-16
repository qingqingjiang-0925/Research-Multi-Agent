"""去重、归一化、合并 reducer 的单元测试。"""

from __future__ import annotations

import pytest

from src.schemas import Document, Stats
from src.state import initial_state, merge_by_id, merge_evidence, merge_stats
from src.utils.dedup import DedupIndex, dedupe_by_key, jaccard, normalize, shingles


class TestNormalize:
    def test_case_and_punct(self):
        assert normalize("Hello, World! 你好，世界。") == "hello world 你好 世界"

    def test_whitespace(self):
        assert normalize("a   b\t\n c") == "a b c"


class TestShinglesJaccard:
    def test_identical_high(self):
        t = "多智能体系统研究综述 2025" * 3
        assert jaccard(shingles(t), shingles(t)) == 1.0

    def test_disjoint_zero(self):
        assert jaccard(shingles("aaaa"), shingles("bbbb")) == 0.0


class TestDedupIndex:
    def test_near_duplicate_detected(self):
        idx = DedupIndex(threshold=0.72)
        base = ("LangGraph 是 LangChain 团队推出的图编排框架，核心抽象是 StateGraph，"
                "节点的状态通道通过 reducer 声明合并策略；" * 4)
        assert idx.add("u1", base)
        # 转载改写：替换个别措辞，相似度仍应超过阈值
        variant = base.replace("推出的", "发布的")
        assert idx.is_duplicate(variant)
        assert not idx.add("u2", variant)

    def test_distinct_not_duplicate(self):
        idx = DedupIndex(threshold=0.72)
        assert idx.add("u1", "多智能体框架对比评测 2025。" * 4)
        assert idx.add("u2", "量子计算纠错码分类综述 2026。" * 4)

    def test_short_text_never_deduped(self):
        idx = DedupIndex(threshold=0.72)
        assert not idx.is_duplicate("短文本")


class TestDedupeByKey:
    def test_keeps_higher_score(self):
        items = [
            {"key": "a", "score": 1},
            {"key": "b", "score": 2},
            {"key": "a", "score": 5},
        ]
        out = dedupe_by_key(items, key_fn=lambda x: x["key"], score_fn=lambda x: x["score"])
        assert [x["score"] for x in out] == [5, 2]


class TestReducers:
    def test_merge_by_id_later_wins(self):
        from src.schemas import SubTask

        left = [SubTask(id="T1", question="a"), SubTask(id="T2", question="b")]
        right = [SubTask(id="T2", question="b2"), SubTask(id="T3", question="c")]
        merged = merge_by_id("id")(left, right)
        assert [t.id for t in merged] == ["T1", "T2", "T3"]
        assert merged[1].question == "b2"

    def test_merge_evidence_prefers_better(self):
        good = Document(id="d1", url="https://a.example/x", score=0.9, content="长正文" * 100)
        weak = Document(id="d1", url="https://a.example/x", score=0.4, snippet="短")
        merged = merge_evidence([weak], [good])
        assert len(merged) == 1
        assert merged[0].score == 0.9

    def test_merge_evidence_preserves_ref(self):
        old = Document(id="d1", url="https://a.example/x", score=0.7, ref=5, content="old" * 50)
        new = Document(id="d1", url="https://a.example/x", score=0.9, ref=0)
        merged = merge_evidence([old], [new])
        assert merged[0].ref == 5  # 已分配的引用编号不被冲掉

    def test_merge_stats_sums(self):
        a = {"llm_calls": 2, "search_calls": 3}
        b = {"llm_calls": 1, "search_calls": 0}
        out = merge_stats(a, b)
        assert out["llm_calls"] == 3
        assert out["search_calls"] == 3
        assert out["prompt_chars"] == 0

    def test_initial_state_shape(self):
        st = initial_state("主题", "上下文", "zh")
        assert st["topic"] == "主题"
        assert st["status"] == "initialized"
        for key in ("evidence", "subtasks", "sections", "drafts", "logs"):
            assert st[key] == []