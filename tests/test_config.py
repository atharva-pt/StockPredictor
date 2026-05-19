from pathlib import Path

import yaml

from trading_copilot.config import load_settings


def test_loads_defaults_when_yaml_missing(tmp_path: Path):
    missing = tmp_path / "nope.yaml"
    s = load_settings(missing)
    assert s.app.name == "trading-copilot"
    assert s.markets.primary == "NSE"


def test_loads_from_yaml(tmp_path: Path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "app": {"log_level": "DEBUG", "timezone": "UTC"},
                "markets": {"primary": "US", "watchlist": {"US": ["AAPL"]}},
                "model": {"horizon_days": 10, "min_confidence": 0.6},
            }
        )
    )
    s = load_settings(cfg)
    assert s.app.log_level == "DEBUG"
    assert s.app.timezone == "UTC"
    assert s.markets.primary == "US"
    assert s.markets.watchlist == {"US": ["AAPL"]}
    assert s.model.horizon_days == 10
    assert s.model.min_confidence == 0.6


def test_paths_are_resolved_absolute():
    s = load_settings()
    assert s.paths.data_dir.is_absolute()
    assert s.paths.log_dir.is_absolute()
    assert s.paths.db_path.parent.exists()  # parent dir was created
