"""Research-Multi-Agent CLI 入口。

用法：
    python main.py demo                                  # 离线全流程演示（无需任何 API key）
    python main.py research "调研主题"                    # 完整调研（需要 .env 配置 LLM）
    python main.py research "主题" --provider mock        # 强制离线
    python main.py chat --thread-id <id>                 # 基于已完成调研继续追问
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Optional

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Research-Multi-Agent：基于 LangGraph 的多智能体深度调研助手",
)


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # 某些嵌入环境不支持 reconfigure
        pass

DEMO_TOPIC = "多智能体深度调研系统：技术方案与主流框架对比"


# --------------------------------------------------------------------------- #
# 环境与公共工具
# --------------------------------------------------------------------------- #
def _apply_env(provider: Optional[str] = None, backend: Optional[str] = None,
               hitl: Optional[bool] = None, log_level: Optional[str] = None,
               scrape: Optional[bool] = None) -> None:
    """CLI 参数 -> 环境变量。必须在 import src.* 之前调用（Settings 读 env）。"""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if backend:
        os.environ["SEARCH_BACKEND"] = backend
    if hitl is not None:
        os.environ["HUMAN_IN_THE_LOOP"] = "true" if hitl else "false"
    if log_level:
        os.environ["LOG_LEVEL"] = log_level
    if scrape is not None:
        os.environ["SCRAPE_ENABLED"] = "true" if scrape else "false"
    try:
        from src.config import get_settings

        get_settings.cache_clear()
    except Exception:  # pragma: no cover
        pass


def _print_result(result: dict) -> None:
    stats = result.get("stats") or {}
    review = result.get("review")
    print("\n" + "=" * 62)
    print(f"📋 报告已生成：{result.get('report_path') or '（未保存）'}")
    print(f"   状态: {result.get('status')} ｜ 证据: {len(result.get('evidence') or [])} 条"
          f" ｜ 章节: {len(result.get('drafts') or [])} 章")
    if review:
        verdict = "✅ 通过" if review.passed else "❌ 未通过"
        print(f"   校验: {verdict}（score={review.score:.2f}，断言 {len(review.claims)} 条）")
    print(f"   成本: LLM {stats.get('llm_calls', 0)} 次 / ~{int((stats.get('prompt_chars', 0) + stats.get('completion_chars', 0)) / 2.5)} tokens"
          f" ｜ 搜索 {stats.get('search_calls', 0)} 次 ｜ 抓取 {stats.get('scrape_calls', 0)} 次")
    print(f"   轮次: 检索 {result.get('search_rounds', 0)} / 修订 {result.get('revision_rounds', 0)}")
    print("=" * 62)


def _run_research(topic: str, context: str, thread_id: str, interactive: bool) -> dict:
    from langgraph.types import Command

    from src.config import get_settings
    from src.graph.builder import build_research_graph
    from src.llm import build_llm
    from src.state import initial_state
    from src.utils.logger import setup_logger
    from src.utils.report import save_state_snapshot

    settings = get_settings()
    setup_logger(settings.log_level)

    try:
        chat_model = build_llm(settings)
    except Exception as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED)
        typer.secho("   提示：复制 .env.example 为 .env 并填入 OPENAI_API_KEY，"
                    "或先用 `python main.py demo` 体验离线流程。", fg=typer.colors.YELLOW)
        raise typer.Exit(1)

    graph = build_research_graph(chat_model, settings)
    config = {"configurable": {"thread_id": thread_id}}
    print(f"🚀 开始调研：{topic}（thread={thread_id}）\n")

    result = graph.invoke(initial_state(topic, context), config=config)
    while "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        value = intr.value if hasattr(intr, "value") else intr
        question = value.get("question", "") if isinstance(value, dict) else str(value)
        print(f"\n🔎 Planner 需要澄清：{question}")
        if interactive:
            answer = input("你的回答（直接回车则按默认假设继续）> ").strip()
        else:
            answer = ""
            print("（非交互模式：按默认假设继续）")
        result = graph.invoke(Command(resume=answer), config=config)

    try:
        snap = save_state_snapshot(result, thread_id, settings.report_dir)
        print(f"💾 对话快照已保存：{snap}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 快照保存失败：{exc}")
    return result


def _chat_repl(thread_id: str, seed: Optional[dict] = None) -> None:
    from src.config import get_settings
    from src.graph.chat import build_chat_graph
    from src.llm import build_llm
    from src.utils.logger import setup_logger

    settings = get_settings()
    setup_logger(settings.log_level)
    try:
        chat_model = build_llm(settings)
    except Exception as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    graph = build_chat_graph(chat_model, settings)
    config = {"configurable": {"thread_id": f"chat-{thread_id}"}}
    seeded = False

    print("\n💬 进入追问模式（基于本次调研的证据回答；输入 exit 退出）")
    while True:
        try:
            question = input("\n追问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("exit", "quit", "q", "退出"):
            break
        payload = {"messages": [{"role": "user", "content": question}]}
        if seed and not seeded:  # 首轮一并注入调研上下文（幂等）
            payload.update({k: v for k, v in seed.items() if k != "messages"})
            seeded = True
        result = graph.invoke(payload, config=config)
        messages = result.get("messages") or []
        for m in reversed(messages):
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            role = (m.get("role") if isinstance(m, dict) else getattr(m, "type", "")) or ""
            if role in ("assistant", "ai"):
                print(f"\n{content}")
                break


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #
@app.command()
def research(
    topic: str = typer.Argument(..., help="调研主题"),
    context: str = typer.Option("", "--context", "-c", help="补充说明：范围 / 时间 / 用途等"),
    provider: str = typer.Option("", "--provider", help="LLM: openai / deepseek / ollama / mock"),
    backend: str = typer.Option("", "--backend", help="搜索: tavily / duckduckgo / mock"),
    thread_id: str = typer.Option("", "--thread-id", help="线程 id（复用 / 续跑）"),
    log_level: str = typer.Option("", "--log-level", help="DEBUG / INFO / WARNING"),
    no_chat: bool = typer.Option(False, "--no-chat", help="完成后不进入追问对话"),
    no_hitl: bool = typer.Option(False, "--no-hitl", help="关闭人工澄清追问"),
) -> None:
    """完整调研：规划 -> 并行检索 -> 写作 -> 校验 -> 报告。"""
    _apply_env(provider or None, backend or None, False if no_hitl else None, log_level or None)
    tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
    result = _run_research(topic, context, tid, interactive=not no_hitl)
    _print_result(result)
    if not no_chat:
        _chat_repl(tid)


@app.command()
def chat(
    thread_id: str = typer.Option("", "--thread-id", help="调研线程 id（对应快照文件）"),
    provider: str = typer.Option("", "--provider", help="LLM: openai / deepseek / ollama / mock"),
    log_level: str = typer.Option("", "--log-level", help="DEBUG / INFO / WARNING"),
) -> None:
    """基于已完成的调研继续追问（读取 reports/ 下的快照）。"""
    _apply_env(provider or None, None, None, log_level or None)
    from src.config import get_settings
    from src.utils.report import load_state_snapshot

    settings = get_settings()
    if not thread_id:
        typer.secho("❌ 请用 --thread-id 指定要追问的调研线程（见 research 输出）。", fg=typer.colors.RED)
        raise typer.Exit(1)
    snap = load_state_snapshot(thread_id, settings.report_dir)
    if snap is None:
        typer.secho(f"❌ 未找到线程 {thread_id} 的快照（{settings.report_dir}）。"
                    "请先运行 research。", fg=typer.colors.RED)
        raise typer.Exit(1)

    seed = {
        "topic": snap.get("topic", ""),
        "evidence": snap.get("evidence") or [],
        "report_markdown": snap.get("report_markdown", ""),
        "messages": [],
    }
    print(f"📚 已加载调研上下文：{snap.get('topic')}（{len(seed['evidence'])} 条证据）")
    _chat_repl(thread_id, seed=seed)


@app.command()
def demo(
    topic: str = typer.Option(DEMO_TOPIC, "--topic", help="演示主题"),
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别"),
) -> None:
    """离线全流程演示：mock LLM + mock 搜索，无需任何 API key。"""
    _apply_env(provider="mock", backend="mock", hitl=False, log_level=log_level, scrape=False)
    tid = f"demo-{uuid.uuid4().hex[:6]}"
    result = _run_research(topic, "", tid, interactive=False)
    _print_result(result)
    print("\n" + "─" * 62)
    print("报告全文：")
    print("─" * 62)
    print(result.get("report_markdown", ""))


if __name__ == "__main__":
    app()
