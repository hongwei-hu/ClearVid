"""Tests for clearvid.app.bootstrap.env_detect."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from clearvid.app.bootstrap.env_detect import GpuInfo, _pick_torch_variant, detect_gpu


# ---------------------------------------------------------------------------
# _pick_torch_variant
# ---------------------------------------------------------------------------

def test_pick_torch_variant_no_driver() -> None:
    url, label = _pick_torch_variant(None)
    assert "cu128" in url


def test_pick_torch_variant_driver_570() -> None:
    url, label = _pick_torch_variant("570.0")
    assert "cu128" in url
    assert "570" in label


def test_pick_torch_variant_driver_560() -> None:
    url, label = _pick_torch_variant("560.0")
    assert "cu126" in url


def test_pick_torch_variant_driver_530() -> None:
    url, label = _pick_torch_variant("530.0")
    assert "cu121" in url


def test_pick_torch_variant_driver_520() -> None:
    url, label = _pick_torch_variant("520.0")
    assert "cu118" in url


def test_pick_torch_variant_old_driver_returns_cpu() -> None:
    url, label = _pick_torch_variant("400.0")
    assert "cpu" in url


def test_pick_torch_variant_invalid_driver_string() -> None:
    """Non-numeric driver string should not crash."""
    url, label = _pick_torch_variant("unknown")
    assert isinstance(url, str)
    assert isinstance(label, str)


# ---------------------------------------------------------------------------
# detect_gpu — mocked subprocess
# ---------------------------------------------------------------------------

def test_detect_gpu_no_nvidia_smi() -> None:
    with patch("clearvid.app.bootstrap.env_detect.shutil.which", return_value=None):
        info = detect_gpu()
    assert info.cuda_capable is False
    assert "cpu" in info.recommended_torch_index.lower()


def test_detect_gpu_subprocess_exception() -> None:
    with patch("clearvid.app.bootstrap.env_detect.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("clearvid.app.bootstrap.env_detect.subprocess.run", side_effect=OSError("boom")):
            info = detect_gpu()
    assert info.cuda_capable is False


def test_detect_gpu_empty_output() -> None:
    mock_result = MagicMock()
    mock_result.stdout = ""
    with patch("clearvid.app.bootstrap.env_detect.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("clearvid.app.bootstrap.env_detect.subprocess.run", return_value=mock_result):
            info = detect_gpu()
    assert info.cuda_capable is False


def test_detect_gpu_valid_output() -> None:
    mock_result = MagicMock()
    mock_result.stdout = "NVIDIA GeForce RTX 4090, 572.16, 24564\n"
    with patch("clearvid.app.bootstrap.env_detect.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("clearvid.app.bootstrap.env_detect.subprocess.run", return_value=mock_result):
            info = detect_gpu()
    assert info.cuda_capable is True
    assert info.name == "NVIDIA GeForce RTX 4090"
    assert info.driver_version == "572.16"
    assert info.memory_mb == 24564


def test_detect_gpu_valid_output_cu128() -> None:
    mock_result = MagicMock()
    mock_result.stdout = "Tesla T4, 575.0, 16160\n"
    with patch("clearvid.app.bootstrap.env_detect.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("clearvid.app.bootstrap.env_detect.subprocess.run", return_value=mock_result):
            info = detect_gpu()
    assert "cu128" in info.recommended_torch_index


# ---------------------------------------------------------------------------
# GpuInfo dataclass defaults
# ---------------------------------------------------------------------------

def test_gpu_info_default_values() -> None:
    info = GpuInfo()
    assert info.name is None
    assert info.cuda_capable is False
    assert info.recommended_torch_index == ""
    assert info.recommended_label == ""
