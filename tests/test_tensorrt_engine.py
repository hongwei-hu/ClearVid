from __future__ import annotations

import time

from clearvid.app.models.tensorrt_engine import (
    _engine_cache_key,
    _onnx_export_shape,
    _read_engine_profile_shapes,
    _score_trt_engine_for_video,
    _trt_build_modes,
    find_compatible_engine,
    list_trt_profile_cache,
    select_compatible_engine_for_video,
    trt_profile_fallbacks,
)
from clearvid.app.gui.workers import TrtWarmupWorker


class _TensorProfileEngine:
    def get_tensor_profile_shape(self, tensor_name: str, profile_index: int):
        assert tensor_name == "input"
        assert profile_index == 0
        return ((1, 3, 128, 128), (4, 3, 512, 512), (4, 3, 544, 544))


class _LegacyProfileEngine:
    def get_profile_shape(self, tensor_name: str, profile_index: int):
        assert tensor_name == "input"
        assert profile_index == 0
        return ((1, 3, 128, 128), (2, 3, 512, 512), (2, 3, 544, 544))


def test_read_engine_profile_shapes_prefers_tensor_api() -> None:
    shapes = _read_engine_profile_shapes(_TensorProfileEngine(), "input", 0)

    assert shapes[1] == (4, 3, 512, 512)
    assert shapes[2][0] == 4


def test_read_engine_profile_shapes_supports_legacy_api() -> None:
    shapes = _read_engine_profile_shapes(_LegacyProfileEngine(), "input", 0)

    assert shapes[1] == (2, 3, 512, 512)
    assert shapes[2][0] == 2


class _NoParamModel:
    def parameters(self):
        return iter(())


def test_find_compatible_engine_prefers_larger_batch_and_tile(tmp_path) -> None:
    model = _NoParamModel()
    weight_path = tmp_path / "model.pth"
    weight_path.write_bytes(b"weights")

    slower = tmp_path / f"realesrgan_{_engine_cache_key(model, True, 512, 4, weight_path)}.engine"
    faster = tmp_path / f"realesrgan_{_engine_cache_key(model, True, 1024, 16, weight_path)}.engine"
    slower.write_bytes(b"engine")
    faster.write_bytes(b"engine")

    assert find_compatible_engine(
        model,
        fp16=True,
        cache_dir=tmp_path,
        weight_path=weight_path,
    ) == (1024, 16, str(faster))


def test_list_trt_profile_cache_reports_ready_failed_and_missing(tmp_path) -> None:
    model = _NoParamModel()
    weight_path = tmp_path / "model.pth"
    weight_path.write_bytes(b"weights")

    ready_digest = _engine_cache_key(model, True, 512, 8, weight_path)
    failed_digest = _engine_cache_key(model, True, 512, 4, weight_path)
    (tmp_path / f"realesrgan_{ready_digest}.engine").write_bytes(b"engine")
    (tmp_path / f"realesrgan_{failed_digest}.failed").write_text(
        f'{{"timestamp": {time.time()}, "reason": "builder failed"}}',
        encoding="utf-8",
    )

    entries = list_trt_profile_cache(
        model,
        fp16=True,
        cache_dir=tmp_path,
        weight_path=weight_path,
        tile_sizes=(512,),
        batch_sizes=(8, 4, 1),
    )
    states = {(entry.tile_size, entry.batch_size): entry.state for entry in entries}

    assert states[(512, 8)] == "ready"
    assert states[(512, 4)] == "failed"
    assert states[(512, 1)] == "missing"
    failed = next(entry for entry in entries if entry.batch_size == 4)
    assert failed.failure_reason == "builder failed"


def test_select_compatible_engine_for_video_prefers_geometry_fit(tmp_path) -> None:
    model = _NoParamModel()
    weight_path = tmp_path / "model.pth"
    weight_path.write_bytes(b"weights")

    for tile_size, batch_size in [(512, 16), (768, 8), (1024, 8)]:
        engine = tmp_path / f"realesrgan_{_engine_cache_key(model, True, tile_size, batch_size, weight_path)}.engine"
        engine.write_bytes(b"engine")

    selected = select_compatible_engine_for_video(
        model,
        width=1280,
        height=720,
        fp16=True,
        cache_dir=tmp_path,
        weight_path=weight_path,
    )

    assert selected is not None
    assert (selected.tile_size, selected.batch_size) == (768, 8)
    assert selected.tiles_per_frame == 2


def test_select_compatible_engine_for_video_skips_profile_min_shape_mismatch(tmp_path) -> None:
    model = _NoParamModel()
    weight_path = tmp_path / "model.pth"
    weight_path.write_bytes(b"weights")

    tiny_invalid = tmp_path / f"realesrgan_{_engine_cache_key(model, True, 1024, 16, weight_path)}.engine"
    tiny_valid = tmp_path / f"realesrgan_{_engine_cache_key(model, True, 512, 16, weight_path)}.engine"
    tiny_invalid.write_bytes(b"engine")
    tiny_valid.write_bytes(b"engine")

    selected = select_compatible_engine_for_video(
        model,
        width=320,
        height=180,
        fp16=True,
        cache_dir=tmp_path,
        weight_path=weight_path,
    )

    assert selected is not None
    assert (selected.tile_size, selected.batch_size) == (512, 16)


def test_score_trt_engine_for_video_rejects_shapes_below_profile_min() -> None:
    assert _score_trt_engine_for_video(
        width=320,
        height=180,
        tile_size=1024,
        batch_size=16,
    ) is None


def test_onnx_export_shape_caps_aggressive_profile() -> None:
    assert _onnx_export_shape(tile_size=1024, batch_size=16) == (1, 512)


def test_trt_profile_fallbacks_move_from_aggressive_to_safe_profiles() -> None:
    assert trt_profile_fallbacks(1024, 16) == [
        (1024, 16),
        (1024, 8),
        (768, 8),
        (512, 8),
        (512, 4),
    ]


def test_trt_build_modes_use_multi_strategy_in_standard_mode() -> None:
    modes = _trt_build_modes(low_load=False)

    assert [mode["label"] for mode in modes] == [
        "standard",
        "high-workspace",
        "compatibility",
    ]
    assert int(modes[1]["workspace_mb"]) > int(modes[0]["workspace_mb"])


def test_trt_build_modes_use_single_strategy_in_low_load_mode() -> None:
    modes = _trt_build_modes(low_load=True)

    assert modes == [
        {
            "label": "low-load",
            "workspace_mb": 256,
            "opt_level": 0,
            "max_aux_streams": 1,
        }
    ]


def test_trt_warmup_detects_recent_failed_status() -> None:
    assert TrtWarmupWorker._is_recent_failed_status("上次部署失败 (2.5 小时前)") is True
    assert TrtWarmupWorker._is_recent_failed_status("TensorRT 引擎尚未部署") is False


def test_trt_failure_summary_identifies_builder_profile_failure() -> None:
    assert TrtWarmupWorker._summarize_trt_failure(
        RuntimeError("TRT 引擎构建失败 (exit=1): ENGINE_BUILD_RETURNED_NONE")
    ) == "TensorRT builder 未能为该 profile 生成 engine"


def test_trt_failure_summary_identifies_all_mode_failure() -> None:
    assert TrtWarmupWorker._summarize_trt_failure(
        RuntimeError("TRT 引擎构建失败 (exit=1): ENGINE_BUILD_RETURNED_NONE_ALL_MODES")
    ) == "TensorRT builder 在当前 profile 的所有构建模式下均未生成 engine"


def test_trt_failure_summary_identifies_cuda_oom() -> None:
    assert TrtWarmupWorker._summarize_trt_failure(
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    ) == "CUDA 显存不足"
