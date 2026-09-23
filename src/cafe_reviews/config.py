"""Config loading. All thresholds, rules, and paths live in configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    path = Path(config_dir) / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data
