from __future__ import annotations

from pathlib import Path

import yaml

from clearvid.app.schemas.models import EnhancementConfig


def load_config(path: Path) -> EnhancementConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EnhancementConfig.model_validate(data)


def save_config(path: Path, config: EnhancementConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(path)
