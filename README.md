# Research-Multi-Agent

基于 **LangGraph** 的多智能体深度调研助手：自动把模糊的调研需求拆解为子问题，多路并行检索，尊重证据地写作，再由独立校验 Agent 复核，最终输出一份**逐句可溯源**的调研报告。

A multi-agent deep-research assistant built on **LangGraph**: decompose → parallel search → evidence-grounded writing → fact-check → a fully cited report.

```
规划(Planner) → 并行检索(Search) → 并行写作(Writer) → 独立校验(Reviewer) → 报告
        └──────────── 证据不足/校验不过 → 定向补检索或重写（有界循环）────────────┘
```

## 为什么是多智能体？

一个调研任务天然可以拆成「拆解 / 检索 / 写作 / 校验」四个正交职责；把每个职责交给一个角色明确的 Agent，能获得四点收益：

| 收益 | 实现 |
| --- | --- |
| **职责边界清晰** | 每个 Agent 用独立 system prompt 约束，Planner 只拆解、Writer 只基于证据、Reviewer 立场是怀疑 |
| **可并行** | 子任务检索、章节写作都用 `Send` 扇出并行执行 |
| **自愈闭环** | 证据不足 / 校验不过时，图自动回到检索或重写节点，而不是把错结果交给用户 |
| **可审计** | Agent 之间传递的是 Pydantic 结构化消息与显式图状态，而不是自由文本对话 |

## 架构

```
src/
├── schemas.py          # Agent 间结构化消息协议（Pydantic）
├── state.py            # LangGraph 状态 + 并行合并 reducer
├── prompts.py          # 集中管理的 prompt（职责边界约束）
├── llm.py              # LLM 接入：structured output 三级降级 + 用量统计
├── mock.py             # 离线 Mock LLM + Mock 搜索（无 key 跑通全流程）
├── agents/
│   ├── planner.py      # 拆解调研需求 → 子任务 + 大纲
│   ├── searcher.py     # 检索子任务：查询改写自愈 + 打分过滤 + 去重
│   ├── writer.py       # 基于证据写作 + 摘要；引用编号确定性校验
│   └── reviewer.py     # 事实校验：claim 判定 + 伪造引用检测 + 评分
├── tools/
│   ├── search.py       # Tavily / DuckDuckGo / Mock 三后端 + 质量打分 + 去重管线
│   └── extractor.py    # 自研网页正文提取（文本密度定位主内容）
├── graph/
│   ├── builder.py      # 主流水线 StateGraph（Send 扇出 / interrupt 澄清）
│   └── chat.py         # 多轮追问图（基于调研成果作答）
└── utils/              # 去重 / 报告渲染 / 日志
```

关键设计点：

- **显式状态 + reducer**：并行分支的写入通过 `merge_by_id` / `merge_evidence` 安全合并，天然幂等。
- **三级结构化输出降级**：tool-calling → JSON-in-prompt 手工解析 → 报错，兼容弱模型。
- **确定性兜底**：引用编号、评分通过条件等全部用代码重算，不信任模型的自我评估。
- **有界循环**：`max_search_rounds=3`、`max_revision_rounds=2`、`max_clarify_rounds=1`，图必然终止。
- **Human-in-the-loop**：需求模糊时 `interrupt()` 暂停追问，回答后原地恢复。

## 快速开始

### 1. 离线演示（无需任何 API key）

```bash
python main.py demo
```

用 Mock LLM + Mock 搜索完整跑一遍「规划 → 并行检索 → 写作 → 校验 → 报告」，报告落在 `reports/`。

### 2. 真实调研

```bash
cp .env.example .env       # 填入 OPENAI_API_KEY（可选 TAVILY_API_KEY）
python main.py research "多智能体框架选型对比" --context "面向中小团队，关注成本与可维护性"
python main.py research "2025 年大模型 Agent 落地现状" --provider deepseek --backend tavily
```

- 只要填了 `OPENAI_API_KEY`（OpenAI 兼容协议，DeepSeek/Qwen/OpenRouter 都可），即可真实运行。
- 没填 `TAVILY_API_KEY` 会自动降级到零 key 的 DuckDuckGo HTML 后端。

### 3. 多轮追问

报告完成后自动进入追问模式，或之后用快照恢复：

```bash
python main.py chat --thread-id demo-1a2b3c
```

追问只基于已检索证据作答，答不了会明说“证据未覆盖”，不编造。

## 配置

所有配置项都可用环境变量 / `.env` 覆盖（字段名即变量名，大小写不敏感）。常用项：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` / `deepseek` / `ollama` / `openrouter` / `mock` |
| `MODEL_NAME` | `gpt-4o-mini` | 模型名 |
| `OPENAI_API_KEY` | 空 | OpenAI 兼容 key |
| `SEARCH_BACKEND` | `tavily` | `tavily` / `duckduckgo` / `mock` |
| `TAVILY_API_KEY` | 空 | Tavily key，缺省自动降级 DDG |
| `MAX_SEARCH_ROUNDS` | `3` | 检索轮次上限 |
| `MAX_REVISION_ROUNDS` | `2` | 修订轮次上限 |
| `REVIEW_PASS_SCORE` | `0.75` | 校验通过阈值 |

## 测试

```bash
pytest
```

覆盖：去重 / 质量打分 / 状态合并 reducer / 报告渲染 / Mock 层，以及端到端的图流程（全流程收敛、补检索与修订闭环、人工澄清 interrupt/resume、追问问答）。全部离线，无网络依赖。

## 目录约定

- `reports/`：生成的报告与追问快照（已 gitignore）
- `.env`：本地密钥（已 gitignore）

## 面试要点速览

1. **分工**：Planner 拆解 → Searcher 检索（含查询改写自愈）→ Writer 尊重证据写作 → Reviewer 事实校验，形成“生成-校验”对抗。
2. **并行**：`Send` 扇出子任务检索与章节写作；`reducer` 让并行写入可安全合并。
3. **反思闭环**：校验发现证据缺口或伪造引用 → 定向补检索 / 只重写 dirty 章节，且有硬上限保证终止。
4. **工程取舍**：结构化输出三级降级、键值去重 + shingle-Jaccard 近重、自研正文提取器（文本密度）、引用编号用代码而不是模型自觉来保证。