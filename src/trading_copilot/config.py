"""Typed configuration loader.

Layered precedence (highest wins):
  1. Environment variables prefixed `TC_` (and values from `.env`)
  2. `config/settings.yaml`
  3. Field defaults below

Usage:
    from trading_copilot.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"


class AppConfig(BaseModel):
    name: str = "trading-copilot"
    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    log_dir: Path = Path("data/logs")
    db_path: Path = Path("data/db/copilot.sqlite")

    def resolved(self, root: Path) -> PathsConfig:
        """Resolve relative paths against the project root."""
        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else (root / p).resolve()

        return PathsConfig(
            data_dir=_abs(self.data_dir),
            cache_dir=_abs(self.cache_dir),
            log_dir=_abs(self.log_dir),
            db_path=_abs(self.db_path),
        )


class MarketsConfig(BaseModel):
    primary: str = "NSE"
    watchlist: dict[str, list[str]] = Field(default_factory=dict)


class DataConfig(BaseModel):
    history_days: int = 730
    default_interval: str = "1d"


class ModelConfig(BaseModel):
    horizon_days: int = 5
    min_confidence: float = 0.55


class Secrets(BaseSettings):
    """Secrets read from environment / .env only — never from YAML."""

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    markets: MarketsConfig = Field(default_factory=MarketsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    secrets: Secrets = Field(default_factory=Secrets)
    project_root: Path = PROJECT_ROOT


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML + env. Pass `config_path` in tests to override."""
    path = config_path or DEFAULT_CONFIG_FILE
    raw = _load_yaml(path)

    settings = Settings(
        app=AppConfig(**raw.get("app", {})),
        paths=PathsConfig(**raw.get("paths", {})).resolved(PROJECT_ROOT),
        markets=MarketsConfig(**raw.get("markets", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        secrets=Secrets(),
    )

    # Make sure runtime directories exist. DB file itself is created by sqlite later.
    for d in (settings.paths.data_dir, settings.paths.cache_dir, settings.paths.log_dir):
        d.mkdir(parents=True, exist_ok=True)
    settings.paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
