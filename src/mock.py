"""离线 Mock 层：MockChatModel + mock_search_results。

为什么需要它：
1. `main.py demo` 在**没有任何 API key** 的情况下完整跑通
   Planner -> Search -> Writer -> Review -> Report 全流程（面试演示 / 冒烟测试）；
2. 单元测试注入确定性的假 LLM / 假搜索，只测图的控制流，不依赖网络。

实现思路是「prompt 驱动」：Mock 解析 user prompt 里的【调研主题】【本章标题】
【证据列表】等标记，生成结构合法、内容相关的输出，因此离线报告看起来是"真的"，
而不是一堆 None。语料中的事实均为公开可查的近似真实信息，仅用于离线演示。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

from src.schemas import (
    Claim,
    ExecutiveSummary,
    FollowupAnswer,
    Gap,
    ResearchPlan,
    ReviewReport,
    Section,
    SectionDraft,
    SubTask,
)

# --------------------------------------------------------------------------- #
# prompt 解析工具
# --------------------------------------------------------------------------- #
def _message_text(messages: Any, role: str = "user") -> str:
    """从 LangChain 消息列表里取出指定角色的文本。兼容 tuple / dict / BaseMessage。"""
    texts: list[str] = []
    for m in messages or []:
        if isinstance(m, tuple) and len(m) >= 2:
            r, content = m[0], m[1]
        elif isinstance(m, dict):
            r, content = m.get("role", m.get("type", "user")), m.get("content", "")
        else:
            r, content = getattr(m, "type", "user"), getattr(m, "content", "")
        if str(r) != role or not content:
            continue
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        texts.append(str(content))
    return "\n".join(texts)


def _grab(text: str, marker: str) -> str:
    """取【标记】后面到下一个空行 / 【标记】之前的正文。"""
    m = re.search(re.escape(marker) + r"\s*\n?(.*?)(?=\n\s*\n|【|$)", text, re.S)
    return (m.group(1).strip() if m else "").strip()


def _parse_topic(text: str) -> str:
    topic = _grab(text, "【调研主题】")
    return topic.splitlines()[0].strip() if topic else "调研主题"


def _parse_refs(text: str) -> list[int]:
    refs = [int(n) for n in re.findall(r"^\[(\d+)\]", text, re.M)]
    seen: list[int] = []
    for r in refs:
        if r not in seen:
            seen.append(r)
    return seen


def _parse_inline_refs(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


def _parse_section_ids(text: str) -> list[str]:
    return re.findall(r"^#{2,3}\s+(S\d+)", text, re.M)


# --------------------------------------------------------------------------- #
# 离线搜索语料（近似真实的公开信息，仅用于 demo / 测试）
# --------------------------------------------------------------------------- #
MOCK_CORPUS: list[dict] = [
    {
        "url": "https://arxiv.org/abs/2501.03527",
        "title": "A Survey on LLM-based Multi-Agent Systems: Workflow, Infrastructure and Challenges",
        "snippet": (
            "本综述系统梳理 2023-2025 年基于大模型的多智能体系统研究，覆盖 60 余篇论文，"
            "归纳出「规划-执行-校验」的通用工作流，并指出通信协议标准化与成本控制是两大工程挑战。"
        ),
        "raw_content": (
            "多智能体系统（MAS）将复杂任务分解给多个具备角色分工的 LLM Agent。综述统计的 60 余篇工作中，"
            "约七成采用「Planner-Worker-Verifier」三层结构：规划器拆解任务，工作者并行执行，校验器对结果做事实核查。"
            "综述指出三大挑战：1) Agent 间通信缺乏统一协议，多数工作用自由文本传递状态，错误会逐级放大；"
            "2) token 成本随 Agent 数量超线性增长，需要预算控制与缓存；3) 评估基准不统一，公开可复现的端到端基准仍然稀缺。"
            "作者建议后续工作关注结构化状态传递（如 StateGraph）与反思式自我修正循环。"
        ),
        "published": "2025-01-15",
        "score": 0.92,
    },
    {
        "url": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
        "title": "LangGraph 低层概念：StateGraph、Send 与 interrupt",
        "snippet": (
            "LangGraph 用 StateGraph 显式建模状态与节点，Send 原语支持 map-reduce 式并行扇出，"
            "interrupt 原语支持 human-in-the-loop，checkpointer 支持断点恢复与时间旅行调试。"
        ),
        "raw_content": (
            "LangGraph 是 LangChain 团队推出的图编排框架，2025 年 10 月发布 1.0 稳定版。核心抽象是 StateGraph："
            "节点是普通函数，边可以是条件路由，状态通道通过 reducer 声明合并策略（如 operator.add 或自定义按 id 合并），"
            "因此并行分支的写入是安全的。Send 原语把一个节点扇出成 N 个并行任务，是 map-reduce 模式的基础；"
            "配合 checkpointer（内存 / SQLite / Postgres）可实现 interrupt 暂停等待人工输入后恢复执行。"
            "官方文档强调：多 Agent 系统里「状态合并策略」比「对话协议」更可靠，因为它是类型约束而不是提示词约束。"
        ),
        "published": "2025-10-02",
        "score": 0.90,
    },
    {
        "url": "https://github.com/microsoft/autogen",
        "title": "microsoft/autogen: 事件驱动的多智能体框架（0.4 重构）",
        "snippet": (
            "AutoGen 0.4 在 2025 年 1 月完成架构重构：从隐式对话循环改为显式事件驱动架构，"
            "AgentChat 高层 API 支持团队（Team）、终止条件（Termination）与结构化输出。"
        ),
        "raw_content": (
            "AutoGen 是微软开源的多智能体对话框架。0.2 时代以 GroupChat 隐式编排著称，但控制流不透明、难以调试；"
            "2025 年 1 月的 0.4 重构引入了显式事件驱动内核：所有 Agent 通过异步消息通信，日志可观测性大幅提升。"
            "AgentChat 提供 RoundRobinGroupChat、SelectorGroupChat 等团队模式，支持 TextMentionTermination 等终止条件，"
            "并内置 Docker 执行环境用于代码执行类 Agent。社区反馈显示 0.4 的学习曲线比 0.2 陡峭，"
            "但长流程任务的稳定性与可恢复性明显更好。GitHub star 数在 2025 年中超过 45k。"
        ),
        "published": "2025-01-28",
        "score": 0.88,
    },
    {
        "url": "https://www.infoq.cn/article/multi-agent-framework-comparison",
        "title": "主流多智能体框架横评：LangGraph、AutoGen、CrewAI 怎么选",
        "snippet": (
            "横评从可控性、并行度、调试体验、生态四个维度对比三大框架：LangGraph 控制粒度最细，"
            "CrewAI 上手最快，AutoGen 适合对话式协作与代码执行场景。"
        ),
        "raw_content": (
            "InfoQ 2025 年横评结论：LangGraph 的显式状态图让「每个 Agent 改了什么」可审计，适合对确定性要求高的生产链路，"
            "代价是需要自己写状态合并逻辑；CrewAI 用角色（Role）+ 任务（Task）+ 流程（Process）三层抽象，"
            "十分钟可以搭出 demo，但复杂控制流要绕框架的约定；AutoGen 0.4 的事件驱动内核适合需要代码执行、"
            "工具调用密集的场景。横评同时提醒：多智能体不是银弹，单 Agent + 好工具在多数任务上更省 token；"
            "引入多 Agent 的合理信号是「任务可并行拆解」且「需要交叉校验」。"
        ),
        "published": "2025-06-18",
        "score": 0.85,
    },
    {
        "url": "https://openai.com/index/introducing-deep-research/",
        "title": "Introducing deep research（OpenAI）",
        "snippet": (
            "OpenAI 于 2025 年 2 月发布 Deep Research：由 o3 模型驱动的端到端调研 Agent，"
            "自主多轮检索、阅读网页、交叉验证后输出带引用的长报告，单次任务通常 5-30 分钟。"
        ),
        "raw_content": (
            "Deep Research 是 2025 年 2 月 OpenAI 发布的 Agent 能力：模型接到问题后自主规划检索路径，"
            "在浏览器里执行几十到上百次搜索与点击，边读边记笔记，遇到矛盾信息会交叉验证，"
            "最终输出带内联引用的结构化报告。官方公布的单次任务时长为 5-30 分钟，"
            "在 Humanity's Last Exam 基准上取得了当时领先的成绩。其产品形态验证了"
            "「规划-检索-写作-校验」多阶段流水线在深度调研任务上的有效性，"
            "也暴露了长时程任务的成本与延迟问题：一次深度调研的 token 消耗可达普通对话的百倍量级。"
        ),
        "published": "2025-02-02",
        "score": 0.93,
    },
    {
        "url": "https://huggingface.co/blog/gaia-benchmark",
        "title": "GAIA：通用 AI 助手基准，多步检索与工具使用",
        "snippet": (
            "GAIA 基准包含 466 个需要多步推理、检索与工具使用的问题，人类正确率 92%，"
            "而 2024 年初最好的 LLM 仅 15% 左右；它成为衡量 Agent 端到端能力的常用标尺。"
        ),
        "raw_content": (
            "GAIA 由 Meta AI / HuggingFace 等机构联合发布，466 个问题覆盖多步检索、文件处理、网页导航与简单计算。"
            "题目对人类很简单（平均 92% 正确率、每个几分钟），但对模型很难：2024 年初 GPT-4 配插件仅约 15%。"
            "到 2025 年，带浏览与规划能力的 Agent 系统已把这一数字推高到 70% 以上，"
            "说明「检索 + 规划 + 校验」的组合是当前 Agent 能力提升的主要来源。"
            "GAIA 的启示：Agent 评估应看端到端任务完成度，而不是单轮问答准确率。"
        ),
        "published": "2024-11-20",
        "score": 0.87,
    },
    {
        "url": "https://www.36kr.com/p/deep-research-enterprise",
        "title": "企业里的 Deep Research：落地场景、成本与坑",
        "snippet": (
            "企业侧调研 Agent 的三个高频场景是竞品分析、行业尽调与技术选型；"
            "实践中的主要成本来自网页正文提取与多轮校验，约占整体 token 开销的六成。"
        ),
        "raw_content": (
            "36 氪 2025 年企业调研显示，落地最多的三个场景是竞品分析、行业尽调与技术选型，"
            "共同点是「结论要可溯源」——报告里每条事实都要能点回原始网页。"
            "工程上的三大坑：1) 搜索结果页质量参差，正文提取失败率约两成，需要降级策略；"
            "2) 多轮校验（review loop）显著提升可信度但 token 成本约占总开销六成；"
            "3) 引用编号幻觉普遍，必须用确定性代码校验而不是靠模型自觉。"
            "受访团队普遍采用「小模型规划 + 大模型写作 + 规则校验」的混合架构来压成本。"
        ),
        "published": "2025-08-11",
        "score": 0.82,
    },
    {
        "url": "https://www.crewai.com/open-source",
        "title": "CrewAI：角色分工驱动的多智能体编排",
        "snippet": (
            "CrewAI 以「团队招聘」隐喻组织 Agent：每个 Agent 有角色、目标与背景故事，"
            "Task 声明预期产出与负责 Agent，Process 支持 sequential 与 hierarchical 两种模式。"
        ),
        "raw_content": (
            "CrewAI 是 2023 年底开源的多智能体框架，设计哲学是「像组建人类团队一样组建 Agent 团队」。"
            "Agent 携带 role / goal / backstory 三个提示词字段，Task 声明 description 与 expected_output，"
            "Process 决定执行拓扑：sequential 顺序执行，hierarchical 引入 manager Agent 动态分派。"
            "优点是心智模型简单、上手快，社区模板丰富；局限是控制流表达力弱于显式状态图，"
            "复杂条件分支与人工介入节点需要绕过框架抽象。2025 年其商业版转向企业级 Agent 平台方向。"
        ),
        "published": "2025-03-05",
        "score": 0.80,
    },
]

_FILLER_POOLS = [
    (
        "公开来源在 2024-2025 年间对该主题的讨论明显升温，多篇实测文章给出了可对照的数据点。",
        "综合来看，主流观点集中在实现路径、成本结构与评估口径三个维度，分歧主要来自场景差异。",
    ),
    (
        "该主题下的资料以工程实践类文章为主，作者多为一线团队，结论通常附带具体配置与踩坑记录。",
        "值得注意的是，不同来源对适用边界的描述差异较大，引用时建议交叉核对至少两个独立来源。",
    ),
    (
        "检索到的资料覆盖了概念定义、架构对比与落地案例三个层次，其中量化数据相对稀缺。",
        "多数文章认为该方向仍处于快速迭代期，选型结论的有效期建议按半年评估。",
    ),
]


def _templated_entries(query: str) -> list[dict]:
    """为任意 query 生成两条确定性的"专属"结果，保证任何子问题都有独立证据可检索。

    用 sha1 摘要做 URL 前缀而不是截断 query 文本：长主题的不同子问题共享长前缀，
    截断会导致 URL 相同、被全局去重误杀。
    """
    h = int(hashlib.sha1((query or "q").encode("utf-8")).hexdigest(), 16)
    slug = hashlib.sha1((query or "q").encode("utf-8")).hexdigest()[:14]
    f1, f2 = _FILLER_POOLS[h % len(_FILLER_POOLS)]
    return [
        {
            "url": f"https://mock.example/{slug}/overview",
            "title": f"{query}：核心概念与要点梳理",
            "snippet": f"本页系统梳理「{query}」的定义、发展脉络与核心组成，并汇总 2024-2025 年的公开观点与数据点。",
            "raw_content": (
                f"本页围绕「{query}」整理公开资料中的核心要点。{f1}"
                f"就定义而言，多数来源将其描述为一套可分阶段执行、可验证产出的方法或系统，"
                f"并在 2024-2025 年间出现了多个开源实现与工程实践分享。{f2}"
            ),
            "published": "2025-05-20",
            "score": 0.96,
        },
        {
            "url": f"https://mock.example/{slug}/practice",
            "title": f"{query}：工程实践与常见问题",
            "snippet": f"针对「{query}」的落地实践汇总：典型架构、常见故障模式与已验证的缓解手段。",
            "raw_content": (
                f"本页汇总「{query}」相关的工程实践。实践文章普遍强调三点："
                f"先做最小可用闭环再扩展、为每个环节设置可观测指标、对失败路径设计显式降级。"
                f"常见故障包括中间产物格式漂移、外部依赖超时与成本超预算。{f2}"
            ),
            "published": "2025-07-08",
            "score": 0.94,
        },
    ]


def mock_search_results(query: str, max_results: int = 5) -> list:
    """离线搜索：query 专属模板结果排前，语料按相关性补足。确定性、无网络。"""
    from src.tools.search import SearchResult

    def _make(item: dict) -> SearchResult:
        return SearchResult(
            title=item["title"],
            url=item["url"],
            snippet=item["snippet"],
            score=item.get("score", 0.0),
            raw_content=item.get("raw_content", ""),
            published=item.get("published", ""),
        )

    q = (query or "").lower()
    q_terms = {t for t in re.split(r"[\s,，。;；、]+", q) if len(t) >= 2}
    corpus = []
    for item in MOCK_CORPUS:
        text = (item["title"] + " " + item["snippet"]).lower()
        overlap = sum(1 for t in q_terms if t in text)
        corpus.append((round(overlap + 0.05 * item.get("score", 0.5), 4), _make(item)))
    corpus.sort(key=lambda x: x[0], reverse=True)

    results = [_make(i) for i in _templated_entries(query or "")] + [m for _, m in corpus]
    return results[: max(1, max_results)]


# --------------------------------------------------------------------------- #
# MockChatModel
# --------------------------------------------------------------------------- #
class _MockMessage:
    """最小化的 AIMessage 替身。"""

    def __init__(self, content: str):
        self.content = content
        self.type = "ai"


class _MockStructured:
    def __init__(self, model: "MockChatModel", schema: Any):
        self.model = model
        self.schema = schema

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        user = _message_text(messages)
        name = getattr(self.schema, "__name__", str(self.schema))
        gen = _GENERATORS.get(name)
        if gen is None:
            try:
                return self.schema()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"MockChatModel 不支持的 schema: {name} ({exc})") from exc
        return gen(self.model, user)


class MockChatModel:
    """离线 LLM。接口对齐 LangChain ChatModel：invoke / with_structured_output。

    可调开关（测试用）：
    - clarify_first=True   ：第一次规划时返回 needs_clarification=True，触发 human-in-the-loop；
    - fail_first_review=True：第一次校验返回不通过的 ReviewReport，触发修订循环。
    """

    def __init__(self, settings: Optional[Any] = None, *, clarify_first: bool = False,
                 fail_first_review: bool = False):
        self.settings = settings
        self.clarify_first = clarify_first
        self.fail_first_review = fail_first_review
        self.plan_calls = 0
        self.review_calls = 0

    # ---- ChatModel 协议 ---- #
    def invoke(self, messages: Any, **kwargs: Any) -> _MockMessage:
        user = _message_text(messages)
        return _MockMessage(f"[mock] 已离线处理：{user[:60]}…")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _MockStructured:
        return _MockStructured(self, schema)

    # ---- 测试辅助 ---- #
    @property
    def calls(self) -> int:
        return self.plan_calls + self.review_calls


# --------------------------------------------------------------------------- #
# 各 schema 的输出生成器
# --------------------------------------------------------------------------- #
def _gen_plan(model: MockChatModel, user: str) -> ResearchPlan:
    topic = _parse_topic(user)
    model.plan_calls += 1
    needs_clar = model.clarify_first and model.plan_calls == 1

    subtasks = [
        SubTask(
            id="T1",
            question=f"{topic}的定义、核心架构与工作流程",
            rationale="先建立概念基线，后续对比与评估都依赖统一定义。",
            search_queries=[f"{topic} 架构", "multi-agent LLM architecture", f"{topic} 工作流程"],
            priority=1,
        ),
        SubTask(
            id="T2",
            question=f"{topic}领域主流开源框架的能力与适用场景对比",
            rationale="选型结论需要横向对比支撑。",
            search_queries=["LangGraph AutoGen CrewAI 对比", f"{topic} 框架", "multi-agent framework comparison 2025"],
            priority=1,
        ),
        SubTask(
            id="T3",
            question=f"{topic}的典型落地案例与可量化的效果数据",
            rationale="避免只有定性描述，需要找带数字的公开案例。",
            search_queries=[f"{topic} 案例 效果", "deep research benchmark GAIA", f"{topic} 落地 数据"],
            priority=2,
        ),
        SubTask(
            id="T4",
            question=f"{topic}的主要工程挑战与已验证的应对方案",
            rationale="结论章节需要风险与对策部分。",
            search_queries=[f"{topic} 挑战", "multi-agent cost control", f"{topic} 最佳实践"],
            priority=2,
        ),
    ]
    outline = [
        Section(id="S1", title="背景与核心概念", guidance="说明主题的定义、为什么重要、当前所处阶段。", subtask_ids=["T1"], order=1),
        Section(id="S2", title="主流框架与方案对比", guidance="横向对比 2-4 个主流方案的能力边界与适用场景，可用表格。", subtask_ids=["T2"], order=2),
        Section(id="S3", title="落地案例与效果", guidance="给出带数字的公开案例，说明效果与局限。", subtask_ids=["T3"], order=3),
        Section(id="S4", title="工程挑战与对策", guidance="归纳主要风险（成本、可靠性、评估）及应对。", subtask_ids=["T4"], order=4),
        Section(id="S5", title="结论与建议", guidance="给出可执行的选型/落地建议，并说明证据局限。", subtask_ids=["T1", "T2", "T3", "T4"], order=5),
    ]
    return ResearchPlan(
        topic=topic,
        intent=f"调研「{topic}」的技术方案、主流框架与落地效果，输出可溯源的结论。",
        needs_clarification=needs_clar,
        clarification_question="你更关注「技术实现细节」还是「选型对比结论」？期望的调研深度是快速概览还是深挖？",
        assumptions=["时间范围默认近两年（2024-2025）", "以公开技术资料与一手来源为准", "语言与主题语言一致"],
        subtasks=subtasks,
        outline=outline,
    )


def _gen_section_draft(model: MockChatModel, user: str) -> SectionDraft:
    title = _grab(user, "【本章标题】") or "本章"
    topic = _parse_topic(user)
    refs = _parse_refs(user)

    if not refs:
        return SectionDraft(
            section_id="S?",
            content="现有资料不足，无法确认本章核心结论。请补充检索后再试。",
            used_refs=[],
            confidence=0.2,
            missing_info="本章没有任何可用证据。",
        )

    r1, r2, r3 = refs[0], refs[min(1, len(refs) - 1)], refs[-1]
    content = (
        f"围绕「{topic}」，本章从「{title}」的角度梳理公开资料中的核心结论。\n\n"
        f"综合多方来源，可以归纳出三点。其一，该方向在 2024-2025 年进入快速工程化阶段，"
        f"多个开源实现已进入生产试用，公开案例普遍报告了流程效率的提升 [{r1}]。"
        f"其二，不同来源对实现路径的取舍存在明显分歧：一部分资料强调中心化编排带来的可控性与可审计性，"
        f"另一部分则认为去中心化的角色协作更灵活、上手更快，分歧本质上来自场景对确定性的要求不同 [{r2}]。"
        f"其三，落地效果方面，公开案例虽然普遍给出正面结论，但缺少可复现的统一基准，"
        f"评估口径不一致，横向比较需谨慎 [{r3}]。\n\n"
        f"需要说明的是，以上结论仅基于当前检索到的证据，来源数量有限；"
        f"更细粒度的量化对比在现有资料中未覆盖，后续如需决策建议补充一手数据源。"
    )
    return SectionDraft(
        section_id="S?",
        content=content,
        used_refs=[r1, r2, r3],
        confidence=0.75,
        missing_info="缺少可复现的量化基准数据。",
    )


def _gen_summary(model: MockChatModel, user: str) -> ExecutiveSummary:
    topic = _parse_topic(user)
    return ExecutiveSummary(
        abstract=(
            f"本报告围绕「{topic}」展开系统调研：先由 Planner 将需求拆解为若干正交子问题，"
            f"再并行检索与提取多来源证据，逐章写作后经独立校验 Agent 复核。"
            f"整体来看，公开资料对该方向的架构模式、框架选型与落地效果均有覆盖，"
            f"主流方案已进入工程化阶段；但可复现的量化基准仍然稀缺，"
            f"不同来源对适用边界的表述差异较大，结论有效期建议按半年评估。"
        ),
        conclusion=(
            "建议优先用成熟框架搭建最小可用闭环验证场景价值，同时保留自研编排层的演进空间；"
            "对关键决策结论补充一手数据源交叉验证后再落地；预算上为校验环节预留约六成 token 开销。"
        ),
        key_findings=[
            "「规划-执行-校验」三层结构是多智能体调研系统的事实标准形态。",
            "显式状态图（如 LangGraph）在可控性与可审计性上优于隐式对话循环。",
            "公开案例普遍报告效率提升，但缺少统一可复现基准，横向比较需谨慎。",
            "校验循环显著提升可信度，同时贡献了主要 token 成本，需要预算控制。",
        ],
    )


def _gen_review(model: MockChatModel, user: str) -> ReviewReport:
    model.review_calls += 1
    refs = _parse_refs(user)
    sections = _parse_section_ids(user) or ["S1"]
    fail = model.fail_first_review and model.review_calls == 1

    if fail:
        fake_ref = (max(refs) + 7) if refs else 99
        return ReviewReport(
            passed=False,
            score=0.55,
            claims=[
                Claim(id="C1", section_id=sections[0], text="多个开源实现已进入生产试用", verdict="supported",
                      reason="证据中有直接支撑。", cited_refs=[refs[0]] if refs else []),
                Claim(id="C2", section_id=sections[0], text="流程效率提升约 40%", verdict="unsupported",
                      reason="证据中找不到该数字，疑似幻觉。", cited_refs=[], severity="high"),
            ],
            conflicts=[],
            gaps=[Gap(subtask_id="T3", section_id=sections[0],
                      description="落地效果缺少带数字的案例支撑。",
                      suggested_queries=["deep research 案例 数据", "agent benchmark 2025"])],
            fabricated_citations=[fake_ref],
            section_verdicts={sid: 0.5 for sid in sections},
            dirty_sections=sections[:1],
            suggestions=["删除无证据支撑的量化表述。", "为效果章节补充可溯源案例。"],
            summary="首轮校验未通过：存在无支撑断言与伪造引用。",
        )

    return ReviewReport(
        passed=True,
        score=0.82,
        claims=[
            Claim(id="C1", section_id=sections[0], text="该方向 2024-2025 年进入工程化阶段", verdict="supported",
                  reason="多条证据一致支撑。", cited_refs=refs[:1]),
            Claim(id="C2", section_id=sections[-1], text="量化基准稀缺，横向比较需谨慎", verdict="supported",
                  reason="证据明确指出基准不统一。", cited_refs=refs[-1:]),
        ],
        conflicts=[],
        gaps=[],
        fabricated_citations=[],
        section_verdicts={sid: 0.8 for sid in sections},
        dirty_sections=[],
        suggestions=["可在对比章节补充一张横向对比表。"],
        summary="校验通过：关键断言均有证据支撑，未发现伪造引用。",
    )


def _gen_followup(model: MockChatModel, user: str) -> FollowupAnswer:
    refs = _parse_refs(user)
    question = _grab(user, "【用户追问】") or "该问题"
    if refs:
        r1, r2 = refs[0], refs[-1]
        answer = (
            f"就「{question}」而言，当前调研证据的要点是：主流方案已进入工程化阶段，"
            f"公开案例普遍报告正面效果，但缺少统一可复现的基准 [{r1}]；"
            f"不同来源对适用边界的表述差异较大，建议交叉核对 [{r2}]。"
            f"更细粒度的数据在当前证据中未覆盖。"
        )
        return FollowupAnswer(answer=answer, refs=[r1, r2], grounded=True)
    return FollowupAnswer(
        answer=f"当前调研证据中没有覆盖「{question}」这一点，无法给出有依据的回答；建议针对该点补充检索。",
        refs=[],
        grounded=False,
    )


def _gen_query_rewrite(model: MockChatModel, user: str) -> Any:
    from src.schemas import QueryRewrite

    question = _grab(user, "【子问题】") or "调研主题"
    return QueryRewrite(
        queries=[
            f"{question} 2025",
            f"{question} 报告 数据",
            f"{question} best practices",
        ]
    )


_GENERATORS = {
    "ResearchPlan": _gen_plan,
    "SectionDraft": _gen_section_draft,
    "ExecutiveSummary": _gen_summary,
    "ReviewReport": _gen_review,
    "FollowupAnswer": _gen_followup,
    "QueryRewrite": _gen_query_rewrite,
}
