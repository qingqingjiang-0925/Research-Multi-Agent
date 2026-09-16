"""测试公共夹具：确保 src.config 在导入时读到 mock/debug 环境。

必须在 `import src.*` 之前执行 —— 本文件顶部就完成了环境变量注入。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("SEARCH_BACKEND", "mock")
os.environ.setdefault("SCRAPE_ENABLED", "false")
os.environ.setdefault("HUMAN_IN_THE_LOOP", "false")
os.environ.setdefault("SAVE_REPORT", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("TEMPERATURE", "0.0")

import pytest  # noqa: E402


@pytest.fixture()
def settings():
    from src.config import get_settings

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture()
def chat_model(settings):
    from src.llm import build_llm

    return build_llm(settings)


@pytest.fixture()
def mock_model():
    from src.mock import MockChatModel

    return MockChatModel()