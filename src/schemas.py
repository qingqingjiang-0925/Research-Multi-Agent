"""Agent 之间传递的结构化数据模型。

用 Pydantic 做 schema 有两个目的：
1. 作为 LLM 的 structured output 约束，避免解析自由文本；
2. 作为 LangGraph 状态里的"消息协议"，让 Agent 之间的接口显式、可测试。
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

SubTaskStatus = Literal["pending", "searching", "done", "gap"]
ClaimVerdict = Literal["supported", "unsupported", "conflicting", "opinion"]


def make_doc_id(url: str) -> str:
    """URL -> 稳定短 id，用于跨轮次去重与引用。"""
    u = re.sub(r"#.*$", "", (url or "").strip()).rstrip("/")
    return hashlib.sha1((u or "unknown").lower().encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class SubTask(BaseModel):
    """一个可独立检索的子问题。"""

    id: str = Field(description="子任务唯一 id，形如 T1 / T2")
    question: str = Field(description="子问题，必须是可以被搜索引擎回答的具体问题")
    rationale: str = Field(default="", description="为什么需要这个子问题")
    search_queries: list[str] = Field(default_factory=list, description="候选搜索关键词，2-3 个")
    priority: int = Field(default=1, ge=1, le=3, description="1 最高")
    status: SubTaskStatus = "pending"
    gap_hint: str = Field(default="", description="上一轮校验发现的证据缺口描述，用于定向补检索")

    def model_post_init(self, __context) -> None:  # noqa: D105
        if not self.search_queries and self.question:
            self.search_queries = [self.question]


class Section(BaseModel):
    """报告大纲中的一个章节。"""

    id: str = Field(description="章节 id，形如 S1 / S2")
    title: str = Field(description="章节标题")
    guidance: str = Field(default="", description="这一章应该写什么、覆盖哪些要点")
    subtask_ids: list[str] = Field(default_factory=list, description="支撑本章的子任务 id")
    order: int = Field(default=0, description="章节顺序，从 1 开始")
    status: Literal["pending", "draft", "dirty", "final"] = "pending"


class ResearchPlan(BaseModel):
    """Planner Agent 的产出。"""

    topic: str = Field(description="规范化后的调研主题")
    intent: str = Field(default="", description="对用户真实意图的一句话理解")
    needs_clarification: bool = Field(default=False, description="需求是否过于模糊、需要追问用户")
    clarification_question: str = Field(default="", description="需要追问时，提一个具体、可一句话回答的问题")
    assumptions: list[str] = Field(default_factory=list, description="在用户未说明时采用的默认假设")
    subtasks: list[SubTask] = Field(default_factory=list)
    outline: list[Section] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
class Document(BaseModel):
    """一条证据（一个网页）。"""

    id: str = Field(description="由 URL 派生的稳定 id")
    url: str
    title: str = ""
    content: str = Field(default="", description="清洗后的网页正文片段")
    snippet: str = Field(default="", description="搜索引擎给出的摘要")
    source: str = Field(default="search", description="tavily / duckduckgo / scrape / mock")
    query: str = Field(default="", description="命中该网页时使用的搜索词")
    subtask_id: str = ""
    score: float = Field(default=0.0, description="质量分 0-1")
    published: str = ""
    ref: int = Field(default=0, description="报告中的参考文献编号，由 collect 节点统一分配")
    scraped: bool = False

    @property
    def best_text(self) -> str:
        return self.content or self.snippet


class SearchOutcome(BaseModel):
    """Search Agent 对单个子任务的执行结果。"""

    subtask_id: str
    documents: list[Document] = Field(default_factory=list)
    queries_used: list[str] = Field(default_factory=list)
    rounds: int = 0
    status: SubTaskStatus = "done"
    note: str = ""


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
class SectionDraft(BaseModel):
    """Writer Agent 产出的单个章节草稿。"""

    section_id: str
    content: str = Field(description="Markdown 正文，事实句末尾带 [n] 引用编号")
    used_refs: list[int] = Field(default_factory=list, description="实际引用到的文献编号")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="证据充分度自评")
    missing_info: str = Field(default="", description="本章缺失、未能证实的信息")
    status: Literal["draft", "final"] = "draft"


class ExecutiveSummary(BaseModel):
    """报告摘要与结论。"""

    abstract: str = Field(default="", description="调研摘要，200-400 字")
    conclusion: str = Field(default="", description="结论与建议")
    key_findings: list[str] = Field(default_factory=list, description="3-6 条关键发现")


# --------------------------------------------------------------------------- #
# Reviewer
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    """一条被核查的事实性陈述。"""

    id: str = Field(description="形如 C1")
    section_id: str = ""
    text: str = Field(description="原文中的陈述（可截断）")
    verdict: ClaimVerdict
    reason: str = Field(default="", description="判定依据")
    cited_refs: list[int] = Field(default_factory=list, description="原文标注的引用编号")
    severity: Literal["low", "medium", "high"] = "low"


class Conflict(BaseModel):
    """多来源信息冲突。"""

    topic: str
    description: str
    refs: list[int] = Field(default_factory=list)


class Gap(BaseModel):
    """证据缺口，会触发 Planner/Search 重新检索。"""

    subtask_id: str = ""
    section_id: str = ""
    description: str = Field(description="缺什么信息")
    suggested_queries: list[str] = Field(default_factory=list, description="建议的补检索关键词")


class ReviewReport(BaseModel):
    """Review Agent 的产出。"""

    passed: bool = Field(default=False, description="是否达到可发布标准")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="整体可信度得分")
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    fabricated_citations: list[int] = Field(default_factory=list, description="引用了不存在的文献编号")
    section_verdicts: dict[str, float] = Field(default_factory=dict, description="每章得分")
    dirty_sections: list[str] = Field(default_factory=list, description="需要重写的章节 id")
    suggestions: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="一句话校验结论")


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
class Stats(BaseModel):
    """成本 / 规模统计，写进报告附录，也用于终止控制。"""

    llm_calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    search_calls: int = 0
    scrape_calls: int = 0
    search_rounds: int = 0
    revision_rounds: int = 0

    @property
    def approx_tokens(self) -> int:
        # 中英混合的粗略估算：约 2.5 字符 / token
        return int((self.prompt_chars + self.completion_chars) / 2.5)

    def merged(self, other: "Stats") -> "Stats":
        data = {k: getattr(self, k) + getattr(other, k) for k in self.model_fields}
        return Stats(**data)


class FollowupAnswer(BaseModel):
    """多轮追问的回答。"""

    answer: str
    refs: list[int] = Field(default_factory=list)
    grounded: bool = Field(default=True, description="是否完全基于已检索证据作答")


class QueryRewrite(BaseModel):
    """Search Agent 内部的查询改写结果。"""

    queries: list[str] = Field(default_factory=list, description="改写后的搜索关键词，短、可直接投给搜索引擎")
