"""Tests for clearvid.app.config."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from clearvid.app.config import load_config, save_config
from clearvid.app.schemas.models import EnhancementConfig, QualityMode, TargetProfile


def _make_config(**kwargs) -> EnhancementConfig:
    defaults = dict(input_path=Path("in.mp4"), output_path=Path("out.mp4"))
    defaults.update(kwargs)
    return EnhancementConfig(**defaults)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_defaults(tmp_path: Path) -> None:
    cfg = _make_config()
    path = tmp_path / "config.yaml"
    save_config(path, cfg)
    loaded = load_config(path)
    assert loaded.target_profile == cfg.target_profile
    assert loaded.quality_mode == cfg.quality_mode
    assert str(loaded.input_path) == str(cfg.input_path)
    assert str(loaded.output_path) == str(cfg.output_path)


def test_save_load_roundtrip_non_default_values(tmp_path: Path) -> None:
    cfg = _make_config(
        target_profile=TargetProfile.SCALE4X,
        quality_mode=QualityMode.FAST,
        preview_seconds=30,
        preserve_audio=False,
        face_restore_enabled=False,
    )
    path = tmp_path / "config.yaml"
    save_config(path, cfg)
    loaded = load_config(path)
    assert loaded.target_profile == TargetProfile.SCALE4X
    assert loaded.quality_mode == QualityMode.FAST
    assert loaded.preview_seconds == 30
    assert loaded.preserve_audio is False
    assert loaded.face_restore_enabled is False


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    cfg = _make_config()
    nested = tmp_path / "a" / "b" / "config.yaml"
    save_config(nested, cfg)
    assert nested.exists()


def test_save_is_valid_yaml(tmp_path: Path) -> None:
    import yaml
    cfg = _make_config(quality_mode=QualityMode.BALANCED)
    path = tmp_path / "config.yaml"
    save_config(path, cfg)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("quality_mode") == "balanced"


def test_save_overwrites_existing_file(tmp_path: Path) -> None:
    cfg1 = _make_config(quality_mode=QualityMode.FAST)
    cfg2 = _make_config(quality_mode=QualityMode.QUALITY)
    path = tmp_path / "config.yaml"
    save_config(path, cfg1)
    save_config(path, cfg2)
    loaded = load_config(path)
    assert loaded.quality_mode == QualityMode.QUALITY


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        load_config(tmp_path / "nonexistent.yaml")
