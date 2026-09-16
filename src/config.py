"""全局配置。

所有配置项都可以用环境变量 / `.env` 覆盖，字段名与环境变量名一致（大小写不敏感）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """运行时配置。"""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- LLM ----------------
    llm_provider: Literal["openai", "deepseek", "ollama", "openrouter", "mock"] = "openai"
    openai_api_base: str = ""
    openai_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    request_timeout: int = 180
    max_retries: int = 2
    ollama_base_url: str = "http://localhost:11434"
    llm_concurrency: int = 4

    # ---------------- 检索 ----------------
    tavily_api_key: str = ""
    search_backend: Literal["tavily", "bing", "duckduckgo", "mock"] = "tavily"
    search_max_results: int = 6
    search_depth: Literal["basic", "advanced"] = "basic"
    search_timeout: int = 20
    bing_market: str = "zh-CN"

    # ---------------- 网页正文提取 ----------------
    scrape_enabled: bool = True
    scrape_timeout: int = 15
    scrape_max_chars: int = 6000
    scrape_concurrency: int = 4

    # ---------------- 流程控制（防死循环 / 控成本） ----------------
    max_search_rounds: int = 3
    max_revision_rounds: int = 2
    max_clarify_rounds: int = 1
    max_subtasks: int = 6
    min_subtasks: int = 3
    min_evidence_per_subtask: int = 2
    max_evidence_per_subtask: int = 6
    max_evidence_total: int = 30
    review_pass_score: float = 0.75
    max_queries_per_round: int = 3

    # ---------------- Human-in-the-loop ----------------
    human_in_the_loop: bool = True

    # ---------------- 输出 ----------------
    log_level: str = "INFO"
    report_dir: Path = Field(default=PROJECT_ROOT / "reports")
    save_report: bool = True

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return (v or "INFO").upper()

    @field_validator("report_dir", mode="before")
    @classmethod
    def _abs_report_dir(cls, v):
        p = Path(v).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    # ---------------- 便捷判断 ----------------
    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key.strip())

    @property
    def has_llm_key(self) -> bool:
        if self.llm_provider in ("mock", "ollama"):
            return True
        return bool(self.openai_api_key.strip())

    def resolved_search_backend(self) -> str:
        """没有 Tavily key 时自动降级到 bing（国内可直连），保证项目开箱可跑。"""
        if self.search_backend == "tavily" and not self.has_tavily:
            return "bing"
        return self.search_backend


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
