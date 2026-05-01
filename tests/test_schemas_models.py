"""Tests for clearvid.app.schemas.models."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from clearvid.app.schemas.models import (
    BackendType,
    BatchResult,
    EnhancementConfig,
    EnvironmentInfo,
    ExecutionPlan,
    FaceRestoreModel,
    HardwareProfile,
    InferenceAccelerator,
    QualityMode,
    StreamInfo,
    TargetProfile,
    UpscaleModel,
    VideoMetadata,
)


# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------

def test_quality_mode_values() -> None:
    assert QualityMode.FAST == "fast"
    assert QualityMode.BALANCED == "balanced"
    assert QualityMode.QUALITY == "quality"


def test_backend_type_values() -> None:
    assert BackendType.AUTO == "auto"
    assert BackendType.BASELINE == "baseline"
    assert BackendType.REALESRGAN == "realesrgan"


def test_target_profile_values() -> None:
    assert TargetProfile.SOURCE == "source"
    assert TargetProfile.FHD == "fhd"
    assert TargetProfile.UHD4K == "uhd4k"
    assert TargetProfile.SCALE2X == "scale2x"
    assert TargetProfile.SCALE4X == "scale4x"


def test_upscale_model_values() -> None:
    assert UpscaleModel.AUTO == "auto"
    assert UpscaleModel.GENERAL_V3 == "general_v3"
    assert UpscaleModel.X4PLUS == "x4plus"


def test_inference_accelerator_values() -> None:
    assert InferenceAccelerator.NONE == "none"
    assert InferenceAccelerator.AUTO == "auto"
    assert InferenceAccelerator.COMPILE == "compile"
    assert InferenceAccelerator.TENSORRT == "tensorrt"


def test_face_restore_model_values() -> None:
    assert FaceRestoreModel.CODEFORMER == "codeformer"
    assert FaceRestoreModel.GFPGAN == "gfpgan"


# ---------------------------------------------------------------------------
# VideoMetadata
# ---------------------------------------------------------------------------

def _make_metadata(**kwargs) -> VideoMetadata:
    defaults = dict(
        input_path=Path("input.mp4"),
        duration_seconds=10.0,
        width=1280,
        height=720,
        fps=30.0,
        video_codec="h264",
    )
    defaults.update(kwargs)
    return VideoMetadata(**defaults)


def test_video_metadata_aspect_ratio() -> None:
    meta = _make_metadata(width=1920, height=1080)
    assert abs(meta.aspect_ratio - (1920 / 1080)) < 1e-9


def test_video_metadata_aspect_ratio_portrait() -> None:
    meta = _make_metadata(width=720, height=1280)
    assert meta.aspect_ratio < 1.0


def test_video_metadata_defaults() -> None:
    meta = _make_metadata()
    assert meta.audio_streams == 0
    assert meta.subtitle_streams == 0
    assert meta.is_interlaced is False
    assert meta.streams == []


def test_stream_info_minimal() -> None:
    s = StreamInfo(index=0, codec_type="video")
    assert s.codec_name is None
    assert s.width is None


# ---------------------------------------------------------------------------
# EnhancementConfig – defaults and validation
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> EnhancementConfig:
    defaults = dict(
        input_path=Path("in.mp4"),
        output_path=Path("out.mp4"),
    )
    defaults.update(kwargs)
    return EnhancementConfig(**defaults)


def test_enhancement_config_defaults() -> None:
    cfg = _make_config()
    assert cfg.target_profile == TargetProfile.FHD
    assert cfg.quality_mode == QualityMode.QUALITY
    assert cfg.backend == BackendType.AUTO
    assert cfg.face_restore_enabled is True
    assert cfg.preserve_audio is True


def test_enhancement_config_face_restore_strength_valid() -> None:
    cfg = _make_config(face_restore_strength=0.0)
    assert cfg.face_restore_strength == 0.0
    cfg2 = _make_config(face_restore_strength=1.0)
    assert cfg2.face_restore_strength == 1.0


def test_enhancement_config_face_restore_strength_invalid() -> None:
    with pytest.raises(ValidationError):
        _make_config(face_restore_strength=-0.1)
    with pytest.raises(ValidationError):
        _make_config(face_restore_strength=1.1)


def test_enhancement_config_temporal_stabilize_strength_invalid() -> None:
    with pytest.raises(ValidationError):
        _make_config(temporal_stabilize_strength=-0.01)
    with pytest.raises(ValidationError):
        _make_config(temporal_stabilize_strength=1.01)


def test_enhancement_config_denoise_strength_invalid() -> None:
    with pytest.raises(ValidationError):
        _make_config(denoise_strength=1.5)


def test_enhancement_config_sharpen_strength_invalid() -> None:
    with pytest.raises(ValidationError):
        _make_config(sharpen_strength=-1.0)


def test_enhancement_config_skip_frame_threshold_invalid() -> None:
    with pytest.raises(ValidationError):
        _make_config(skip_frame_threshold=-1.0)
    with pytest.raises(ValidationError):
        _make_config(skip_frame_threshold=21.0)


def test_enhancement_config_preprocess_deinterlace_invalid() -> None:
    with pytest.raises(ValidationError):
        _make_config(preprocess_deinterlace="force")


def test_enhancement_config_roundtrip_json() -> None:
    cfg = _make_config(
        target_profile=TargetProfile.SCALE4X,
        quality_mode=QualityMode.BALANCED,
        preview_seconds=5,
    )
    data = cfg.model_dump(mode="json")
    cfg2 = EnhancementConfig.model_validate(data)
    assert cfg2.target_profile == TargetProfile.SCALE4X
    assert cfg2.quality_mode == QualityMode.BALANCED
    assert cfg2.preview_seconds == 5


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------

def test_execution_plan_defaults() -> None:
    plan = ExecutionPlan(output_width=1920, output_height=1080, backend=BackendType.REALESRGAN)
    assert plan.command == []
    assert plan.notes == []


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------

def test_batch_result_success() -> None:
    r = BatchResult(
        input_path=Path("in.mp4"),
        output_path=Path("out.mp4"),
        success=True,
        message="OK",
        backend=BackendType.REALESRGAN,
    )
    assert r.success is True
    assert r.backend == BackendType.REALESRGAN


def test_batch_result_failure_no_backend() -> None:
    r = BatchResult(
        input_path=Path("in.mp4"),
        output_path=Path("out.mp4"),
        success=False,
        message="error",
    )
    assert r.backend is None


# ---------------------------------------------------------------------------
# EnvironmentInfo
# ---------------------------------------------------------------------------

def test_environment_info_defaults() -> None:
    env = EnvironmentInfo(ffmpeg_available=True, ffprobe_available=True, nvidia_smi_available=False)
    assert env.preferred_backend == BackendType.BASELINE
    assert env.realesrgan_available is False
    assert env.torch_cuda_available is False
    assert env.ffmpeg_hwaccels == []
    assert env.ffmpeg_encoders == []
