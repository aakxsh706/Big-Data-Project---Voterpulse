"""Centralised configuration loading.

All YAML files in ``config/`` are parsed into validated Pydantic models once per
process. Environment variables override the few values that differ per machine.
Nothing in the codebase should read YAML or environment variables directly.
"""

from __future__ import annotations

import os
from datetime import timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# India has used a fixed UTC+05:30 offset with no daylight saving since 1945, so a
# fixed offset is exact and avoids needing tzdata on Windows.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")


# --------------------------------------------------------------------------- models
class IssueDefinition(BaseModel):
    key: str
    name: str
    short_name: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)


class PromptExample(BaseModel):
    headline: str
    issue: str
    evidence: str
    secondary: str | None = None
    note: str | None = None


class IssuesConfig(BaseModel):
    not_relevant_label: str
    issues: list[IssueDefinition]
    prompt_examples: list[PromptExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "IssuesConfig":
        names = [i.name for i in self.issues]
        if len(names) != len(set(names)):
            raise ValueError("issues.yaml: issue names must be unique")
        keys = [i.key for i in self.issues]
        if len(keys) != len(set(keys)):
            raise ValueError("issues.yaml: issue keys must be unique")
        if self.not_relevant_label in names:
            raise ValueError("issues.yaml: not_relevant_label clashes with an issue name")
        allowed = set(names) | {self.not_relevant_label}
        for ex in self.prompt_examples:
            if ex.issue not in allowed:
                raise ValueError(f"issues.yaml: prompt example uses unknown issue {ex.issue!r}")
        return self

    @property
    def issue_names(self) -> list[str]:
        return [i.name for i in self.issues]

    @property
    def all_labels(self) -> list[str]:
        return self.issue_names + [self.not_relevant_label]

    def by_key(self, key: str) -> IssueDefinition:
        for issue in self.issues:
            if issue.key == key:
                return issue
        raise KeyError(key)

    def by_name(self, name: str) -> IssueDefinition:
        for issue in self.issues:
            if issue.name == name:
                return issue
        raise KeyError(name)


class FeedDefinition(BaseModel):
    key: str
    query: str


class RssRequestConfig(BaseModel):
    timeout_seconds: float = 15
    max_retries: int = 3
    backoff_seconds: float = 2
    pause_between_feeds_seconds: float = 2
    user_agent: str = "VoterPulseLite/1.0"


class RssConfig(BaseModel):
    base_url: str
    params: dict[str, str]
    request: RssRequestConfig = Field(default_factory=RssRequestConfig)
    feeds: list[FeedDefinition]


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    expected_digest: str | None = None
    timeout_seconds: float = 60
    max_retries: int = 2
    backoff_seconds: float = 1.0
    backoff_max_seconds: float = 8.0
    keep_alive: str = "15m"


class ModelConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    options: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0})


class HeadlineThresholds(BaseModel):
    positive_min: float = 0.05
    negative_max: float = -0.05

    @model_validator(mode="after")
    def _check(self) -> "HeadlineThresholds":
        if not self.negative_max < self.positive_min:
            raise ValueError("headline thresholds: negative_max must be below positive_min")
        return self


class AggregateThresholds(BaseModel):
    positive_mean: float = 0.10
    negative_mean: float = -0.10
    share_margin: float = 0.10
    neutral_dominant_share: float = 0.60


class TrendThresholds(BaseModel):
    change_min: float = 0.05
    min_headlines_each_window: int = 25


class SentimentThresholds(BaseModel):
    headline: HeadlineThresholds = Field(default_factory=HeadlineThresholds)
    aggregate: AggregateThresholds = Field(default_factory=AggregateThresholds)
    trend: TrendThresholds = Field(default_factory=TrendThresholds)


class DatabaseSettings(BaseModel):
    path: str = "data/voterpulse.db"
    demo_path: str = "data/demo/voterpulse_demo.db"
    busy_timeout_ms: int = 10000


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str | None = "data/logs/voterpulse.log"
    max_bytes: int = 2_000_000
    backup_count: int = 3


class PipelineSettings(BaseModel):
    interval_minutes: int = 15
    max_classifications_per_run: int = 120
    stale_run_minutes: int = 30
    llm_circuit_breaker_failures: int = 3


class AggregationSettings(BaseModel):
    min_headlines: int = 25
    stale_after_minutes: int = 60
    recent_headlines_limit: int = 50


class AlertSettings(BaseModel):
    current_window_hours: int = 24
    reference_window_days: int = 7
    min_current_headlines: int = 25
    min_reference_headlines: int = 25
    trigger_delta: float = 0.15
    large_delta: float = 0.30
    reset_delta: float = 0.075

    @model_validator(mode="after")
    def _check(self) -> "AlertSettings":
        if not 0 <= self.reset_delta < self.trigger_delta <= self.large_delta:
            raise ValueError("alerts: need 0 <= reset_delta < trigger_delta <= large_delta")
        return self


class Settings(BaseModel):
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    demo_enabled: bool = False
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    aggregation: AggregationSettings = Field(default_factory=AggregationSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)


class AppConfig(BaseModel):
    """Everything the application needs, validated."""

    settings: Settings
    issues: IssuesConfig
    rss: RssConfig
    model: ModelConfig
    thresholds: SentimentThresholds

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def db_path(self) -> Path:
        """Database in use: the demo database when demo mode is on, else the live one."""
        db = self.settings.database
        return self.resolve_path(db.demo_path if self.settings.demo_enabled else db.path)

    @property
    def live_db_path(self) -> Path:
        return self.resolve_path(self.settings.database.path)

    @property
    def demo_db_path(self) -> Path:
        return self.resolve_path(self.settings.database.demo_path)


# --------------------------------------------------------------------------- loading
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping at the top level")
    return data


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(path: Path) -> None:
    """Minimal .env support (KEY=VALUE lines). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(config_dir: Path | None = None, env: dict[str, str] | None = None) -> AppConfig:
    """Load and validate all configuration. ``env`` defaults to ``os.environ``."""
    config_dir = config_dir or CONFIG_DIR
    if env is None:
        _load_dotenv(PROJECT_ROOT / ".env")
        env = dict(os.environ)

    raw_settings = _read_yaml(config_dir / "settings.yaml")
    settings = Settings(
        database=DatabaseSettings(**raw_settings.get("database", {})),
        demo_enabled=bool(raw_settings.get("demo", {}).get("enabled", False)),
        logging=LoggingSettings(**raw_settings.get("logging", {})),
        pipeline=PipelineSettings(**raw_settings.get("pipeline", {})),
        aggregation=AggregationSettings(**raw_settings.get("aggregation", {})),
        alerts=AlertSettings(**raw_settings.get("alerts", {})),
    )
    model = ModelConfig(**_read_yaml(config_dir / "model.yaml"))

    if env.get("VOTERPULSE_DB_PATH"):
        settings.database.path = env["VOTERPULSE_DB_PATH"]
    if env.get("VOTERPULSE_DEMO_MODE"):
        settings.demo_enabled = _env_bool(env["VOTERPULSE_DEMO_MODE"])
    if env.get("VOTERPULSE_LOG_LEVEL"):
        settings.logging.level = env["VOTERPULSE_LOG_LEVEL"].upper()
    if env.get("VOTERPULSE_OLLAMA_HOST"):
        model.ollama.host = env["VOTERPULSE_OLLAMA_HOST"]
    if env.get("VOTERPULSE_MODEL"):
        model.ollama.model = env["VOTERPULSE_MODEL"]

    return AppConfig(
        settings=settings,
        issues=IssuesConfig(**_read_yaml(config_dir / "issues.yaml")),
        rss=RssConfig(**_read_yaml(config_dir / "rss_feeds.yaml")),
        model=model,
        thresholds=SentimentThresholds(**_read_yaml(config_dir / "sentiment_thresholds.yaml")),
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached configuration."""
    return load_config()
