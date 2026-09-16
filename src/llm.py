"""LLM 接入层。

- `build_llm`：按 provider 构造 LangChain Chat 模型（OpenAI 兼容协议覆盖 DeepSeek / Qwen / OpenRouter）。
- `LLMClient`：统一封装 structured output + 文本输出 + 失败降级 + 用量统计。
  所有 Agent 只依赖 `LLMClient`，因此可以在测试里整体替换成 Mock。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import Settings, get_settings
from src.schemas import Stats
from src.utils.logger import get_logger

log = get_logger("llm")

T = TypeVar("T", bound=BaseModel)

OPENAI_COMPAT_BASE = {
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

DEFAULT_MODEL = {
    "deepseek": "deepseek-chat",
    "openrouter": "openai/gpt-4o-mini",
    "ollama": "qwen2.5:7b",
}


class _Structured(Protocol):
    def invoke(self, messages: Any, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    def with_structured_output(self, schema: Any, **kwargs: Any) -> _Structured: ...
    def invoke(self, messages: Any, **kwargs: Any) -> Any: ...


def build_llm(settings: Optional[Settings] = None, *, provider: Optional[str] = None,
              model: Optional[str] = None, temperature: Optional[float] = None) -> Any:
    """构造底层 chat model。provider=mock 时返回 MockChatModel（见 src/mock.py）。"""
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.model_name
    temperature = settings.temperature if temperature is None else temperature

    if provider == "mock":
        from src.mock import MockChatModel

        return MockChatModel(settings=settings)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or DEFAULT_MODEL["ollama"],
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI

    base_url = settings.openai_api_base or OPENAI_COMPAT_BASE.get(provider, "")
    if not settings.openai_api_key:
        raise RuntimeError(
            f"缺少 OPENAI_API_KEY（provider={provider}）。请在 .env 中配置，或用 --provider mock 跑离线演示。"
        )
    return ChatOpenAI(
        model=model or DEFAULT_MODEL.get(provider, "gpt-4o-mini"),
        api_key=settings.openai_api_key,
        base_url=base_url or None,
        temperature=temperature,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def extract_json(text: str) -> str:
    """从自由文本里抠出第一个完整 JSON 对象/数组。"""
    if not text:
        return ""
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        return m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts)
    return str(content or "")


class LLMClient:
    """对 Agent 暴露的唯一 LLM 接口。"""

    def __init__(self, llm: Any, settings: Optional[Settings] = None):
        self.llm = llm
        self.settings = settings or get_settings()
        self.stats = Stats()

    # ------------------------------------------------------------------ #
    def _messages(self, system: str, user: str) -> list[tuple[str, str]]:
        return [("system", system), ("user", user)]

    def _account(self, messages: Any, output_text: str) -> None:
        self.stats.llm_calls += 1
        self.stats.prompt_chars += sum(len(_content_to_text(m[1] if isinstance(m, tuple) else m)) for m in messages)
        self.stats.completion_chars += len(output_text or "")

    # ------------------------------------------------------------------ #
    def structured(self, schema: type[T], system: str, user: str, *, task: str = "",
                   retries: int = 2) -> T:
        """结构化输出。

        三级降级：function-calling/json_schema -> 提示词要求 JSON + 手工解析 -> 抛错。
        小模型经常不支持 tool calling，这个降级链是保证项目能跑在不同模型上的关键。
        """
        messages = self._messages(system, user)
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                runner = self.llm.with_structured_output(schema)
                result = runner.invoke(messages)
                if isinstance(result, schema):
                    self._account(messages, result.model_dump_json())
                    return result
                if isinstance(result, dict):
                    obj = schema.model_validate(result)
                    self._account(messages, obj.model_dump_json())
                    return obj
                if result is None:
                    raise ValueError("structured output 返回 None")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.debug(f"[{task or schema.__name__}] structured 第 {attempt + 1} 次失败: {exc}")

        # 降级：让模型直接吐 JSON
        json_hint = (
            "\n\n【输出格式】只输出一个 JSON 对象，不要 markdown 代码围栏，不要任何解释文字。"
            f"JSON 必须符合以下 schema：\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)[:4000]}"
        )
        for attempt in range(retries + 1):
            try:
                resp = self.llm.invoke(self._messages(system + json_hint, user))
                text = _content_to_text(getattr(resp, "content", resp))
                self._account(messages, text)
                obj = schema.model_validate(json.loads(extract_json(text)))
                return obj
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                log.warning(f"[{task or schema.__name__}] JSON 降级解析第 {attempt + 1} 次失败: {exc}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning(f"[{task or schema.__name__}] LLM 调用失败: {exc}")

        raise RuntimeError(f"LLM 结构化输出失败（task={task or schema.__name__}）: {last_error}")

    # ------------------------------------------------------------------ #
    def text(self, system: str, user: str, *, task: str = "") -> str:
        messages = self._messages(system, user)
        resp = self.llm.invoke(messages)
        text = _content_to_text(getattr(resp, "content", resp)).strip()
        self._account(messages, text)
        return text

    def dump_stats(self) -> dict:
        return self.stats.model_dump()

    def usage_delta(self, before: dict) -> dict:
        """返回相对快照 `before` 的增量。只含 LLM 侧字段，避免覆盖搜索侧计数。"""
        after = self.stats.model_dump()
        keys = ("llm_calls", "prompt_chars", "completion_chars")
        return {k: after.get(k, 0) - before.get(k, 0) for k in keys}
