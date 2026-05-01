from __future__ import annotations

import logging
import math
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

import cv2
import numpy as np

from clearvid.app.models import stream_codec
from clearvid.app.models.codeformer_runner import CodeFormerRestorer
from clearvid.app.models.gfpgan_runner import GFPGANRestorer
from clearvid.app.models.perf_diagnostics import (
    CpuUsageTracker,
    GpuSampler,
    format_gpu_snapshot,
    format_gpu_summary,
    format_queue_info,
)
from clearvid.app.models.tensorrt_engine import (
    InferenceAccelerator,
    TRT_MAX_BATCH,
    accelerate_model,
    check_engine_ready,
    describe_accelerator,
    select_compatible_engine_for_video,
)
from clearvid.app.models.realesrgan_support import (
    DEFAULT_MODEL_FILENAME,
    DEFAULT_MODEL_SCALE,
    DEFAULT_MODEL_URL,
    FrameSkipper,
    _MODEL_REGISTRY,
    _apply_quality_mode_overrides,
    _auto_batch_size,
    _auto_trt_tile_size,
    _build_codeformer_restorer,
    _build_decode_command,
    _build_encode_command,
    _build_temporal_stabilizer,
    _build_upsampler,
    _estimate_total_frames,
    _fit_and_pad_frame,
    _load_runtime_components,
    _map_frame_progress,
    _mux_output,
    _mux_preview,
    _resolve_accelerator,
    _resolve_outscale,
    _resolve_tile_size,
    _resolve_trt_batch,
    _resize_for_target,
    ensure_realesrgan_weights,
    find_realesrgan_weights,
    inspect_realesrgan_runtime,
    resolve_upscale_model,
    validate_realesrgan_environment,
)
from clearvid.app.models.realesrgan_streaming import (
    _PendingTrtFrameBatch,
    _cleanup_stream_processes,
    _collect_batch,
    _copy_trt_frames_to_cpu,
    _emit_progress,
    _enhance_frame_trt_tiled,
    _enhance_frames_batch,
    _enhance_frames_trt_tiled,
    _fetch_enhanced_frames,
    _finalize_stream_processes,
    _flush_trt_tile_batch,
    _frame_payload_count,
    _is_trt_upsampler,
    _iter_frame_payload,
    _pack_trt_output_tensor,
    _pad_trt_tile_to_shape,
    _prepare_encoder_frame,
    _prepare_encoder_frame_batch,
    _process_stream_frames,
    _read_exact_bytes,
    _record_and_queue_enhanced_payload,
    _report_stream_progress,
    _resolve_frame_payload,
    _start_decode_thread,
    _trt_output_to_frame,
    _trt_output_to_frames,
    _try_async_trt_cpu_transfer,
    _write_encoder_frame,
    _write_encoder_frame_batch,
    _write_enhanced_frames,
    _write_finalized_frames,
)
from clearvid.app.bootstrap.paths import (
    REALESRGAN_WEIGHTS_DIR,
    TRT_CACHE_DIR,
    WEIGHTS_DIR,
)
from clearvid.app.postprocess.enhance import apply_sharpening
from clearvid.app.postprocess.temporal_stabilizer import TemporalStabilizer
from clearvid.app.schemas.models import (
    EnhancementConfig,
    FaceRestoreModel,
    InferenceAccelerator as InferenceAcceleratorEnum,
    QualityMode,
    TargetProfile,
    UpscaleModel,
    VideoMetadata,
)
from clearvid.app.export_control import ExportCancelled, ExportControl
from clearvid.app.utils.subprocess_utils import run_command

def run_realesrgan_video(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    progress_callback: Callable[[int, str], None] | None = None,
    control: ExportControl | None = None,
    preview_callback: Callable[[str], None] | None = None,
) -> None:
    config = _apply_quality_mode_overrides(config)
    t_start = time.perf_counter()

    # Resolve accelerator FIRST so model selection can consider it
    accel = _resolve_accelerator(config.inference_accelerator)
    accel_label = describe_accelerator(accel)

    _emit_progress(progress_callback, 6, "正在准备 Real-ESRGAN 权重")
    weights_dir = REALESRGAN_WEIGHTS_DIR
    model_key = resolve_upscale_model(config.upscale_model, config.quality_mode, accel)
    model_arch = _MODEL_REGISTRY[model_key]["arch"]
    config = config.model_copy(update={
        "batch_size": _auto_batch_size(config, metadata.width, metadata.height, model_arch),
    })
    model_path = ensure_realesrgan_weights(weights_dir, model_key)
    model_label = _MODEL_REGISTRY[model_key]["filename"]
    _emit_progress(progress_callback, 10, f"正在初始化 Real-ESRGAN ({model_label})")
    upsampler = cast(Any, _build_upsampler(config, model_path, model_key, metadata.width, metadata.height))
    tile_size = getattr(upsampler, "tile_size", getattr(upsampler, "tile", "?"))
    if accel != InferenceAccelerator.NONE:
        _emit_progress(progress_callback, 11, f"正在应用推理加速: {accel_label}")
        t_accel = time.perf_counter()
        trt_tile = _auto_trt_tile_size(config.tile_size, metadata.width, metadata.height)
        trt_batch = _resolve_trt_batch(config.batch_size, accel, config.inference_accelerator)
        build_if_missing = False

        if accel == InferenceAccelerator.TENSORRT:
            ready, _msg = check_engine_ready(
                upsampler.model,
                fp16=config.fp16_enabled,
                tile_size=trt_tile,
                batch_size=trt_batch,
                cache_dir=TRT_CACHE_DIR,
                weight_path=model_path,
            )
            selected = None
            should_auto_select = (
                config.inference_accelerator.value == "auto"
                or config.tile_size <= 0
                or config.batch_size <= 0
                or not ready
            )
            if should_auto_select:
                selected = select_compatible_engine_for_video(
                    upsampler.model,
                    width=metadata.width,
                    height=metadata.height,
                    tile_pad=config.tile_pad,
                    fp16=config.fp16_enabled,
                    cache_dir=TRT_CACHE_DIR,
                    weight_path=model_path,
                )

            if selected is not None:
                if selected.tile_size != trt_tile or selected.batch_size != trt_batch:
                    logger.info(
                        "自适应选择 TensorRT 引擎: requested tile=%d batch=%d -> selected tile=%d batch=%d "
                        "(tiles/frame=%d score=%.0f)",
                        trt_tile, trt_batch,
                        selected.tile_size, selected.batch_size,
                        selected.tiles_per_frame, selected.score,
                    )
                    _emit_progress(
                        progress_callback,
                        11,
                        f"自适应选择 TensorRT 引擎: tile={selected.tile_size} batch={selected.batch_size} "
                        f"({selected.tiles_per_frame} tiles/帧)",
                    )
                trt_tile = selected.tile_size
                trt_batch = selected.batch_size
                config = config.model_copy(update={"tile_size": trt_tile, "batch_size": trt_batch})
            elif ready:
                config = config.model_copy(update={"tile_size": trt_tile, "batch_size": trt_batch})
            elif config.inference_accelerator.value != "auto":
                raise RuntimeError(
                    f"TensorRT 引擎尚未部署 (tile={trt_tile}, batch={trt_batch})。"
                    "请先使用 GUI 中的'部署 TensorRT 引擎'按钮 "
                    "或运行 `clearvid warmup` 命令完成首次构建。\n"
                    "或切换到'自动检测'模式以自动选择可用加速方案。"
                )
            else:
                logger.info("自动模式未找到可用 TensorRT 缓存引擎，降级到 torch.compile/标准推理")
                _emit_progress(
                    progress_callback,
                    11,
                    "未找到可复用 TensorRT 引擎，自动降级到 torch.compile/标准推理",
                )
                accel = InferenceAccelerator.COMPILE

        upsampler.model = accelerate_model(
            upsampler.model,
            accel,
            fp16=config.fp16_enabled,
            tile_size=trt_tile,
            batch_size=trt_batch,
            cache_dir=TRT_CACHE_DIR,
            progress_callback=progress_callback,
            weight_path=model_path,
            trt_build_timeout=getattr(config, 'trt_build_timeout', None),
            build_if_missing=build_if_missing,
            low_load=True,
        )
        accel_actual = "TensorRT" if hasattr(upsampler.model, '_engine') else "PyTorch (回退)"
        # When TRT is active, force the upsampler tile to the selected profile.
        # Whole-frame mode or stale GUI values can send shapes outside profile bounds.
        if hasattr(upsampler.model, '_engine'):
            upsampler.tile = trt_tile
            upsampler.tile_size = trt_tile
            logger.info("TRT 激活: 使用 tiling=%d batch=%d", trt_tile, trt_batch)

        # Performance advisory: TRT batch=1 with large free VRAM is GPU-underutilised.
        # Tile inference is sequential — each TRT call covers one tile, leaving GPU
        # idle between kernel launches.  batch=4+ in the engine profile would allow
        # multi-tile fusing for ~3-4× better throughput.
        if hasattr(upsampler.model, '_engine') and trt_batch == 1:
            try:
                import torch
                if torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info(0)
                    if free_bytes > 8 * 1024 ** 3:
                        logger.warning(
                            "TRT batch=1 性能提示: GPU 剩余显存 %.0fGB，当前引擎每次仅处理 1 个 tile，"
                            "GPU 利用率偏低。建议点击『部署 TRT 引擎』按钮重新部署 (batch=4) 以获得 3-4× 速度提升；"
                            "或在『加速方式』下拉菜单切换为『自动检测』改用 PyTorch 批量模式，无需重新部署。",
                            free_bytes / 1024 ** 3,
                        )
            except Exception:  # noqa: BLE001
                pass

        _emit_progress(
            progress_callback, 11,
            f"推理加速就绪: {accel_actual} ({time.perf_counter() - t_accel:.1f}s)",
        )

    _emit_progress(progress_callback, 12, "正在初始化人脸修复")
    codeformer_restorer = _build_codeformer_restorer(config, metadata, output_width, output_height)
    _emit_progress(progress_callback, 14, "正在初始化时序稳定器")
    stabilizer = _build_temporal_stabilizer(config)

    # Log pipeline configuration summary
    t_init = time.perf_counter() - t_start
    actual_tile = getattr(upsampler, 'tile', tile_size)
    batching = "整帧" if actual_tile == 0 else f"tile={actual_tile}"
    face_label = "关闭" if codeformer_restorer is None else type(codeformer_restorer).__name__
    _emit_progress(
        progress_callback, 15,
        f"初始化完成 ({t_init:.1f}s) | 模型={model_label} 加速={accel_label} "
        f"batch={config.batch_size} {batching} fp16={'是' if config.fp16_enabled else '否'} "
        f"人脸={face_label} {'async' if config.async_pipeline else 'sync'}",
    )

    # Emit structured task config block for diagnostics
    fps_src = metadata.fps or 0.0
    dur_s = metadata.duration_seconds or 0.0
    total_frames_est = int(dur_s * fps_src) if dur_s > 0 and fps_src > 0 else 0
    dur_str = f"{int(dur_s // 60)}m{int(dur_s % 60)}s" if dur_s > 0 else "未知"
    stabilizer_label = "开启" if config.temporal_stabilize_enabled else "关闭"
    sharpen_label = f"{config.sharpen_strength:.1f}" if config.sharpen_enabled else "关闭"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _gpu_name = _torch.cuda.get_device_properties(0).name
            _vram_total_gb = _torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            gpu_line = f"{_gpu_name} ({_vram_total_gb:.0f}GB VRAM)"
        else:
            gpu_line = "未检测到 CUDA GPU"
    except Exception:  # noqa: BLE001
        gpu_line = "未知"
    _emit_progress(
        progress_callback, 15,
        f"[任务配置] "
        f"输入: {metadata.width}×{metadata.height} {fps_src:.3f}fps ~{total_frames_est or '?'}帧({dur_str}) | "
        f"输出: {output_width}×{output_height} 缩放×{output_width // max(metadata.width, 1)} | "
        f"模型: {model_label} | 加速: {accel_label} batch={config.batch_size} {batching} fp16={'是' if config.fp16_enabled else '否'} | "
        f"后处理: 人脸={face_label} 稳定={stabilizer_label} 锐化={sharpen_label} | "
        f"GPU: {gpu_line}",
    )

    temp_root = Path(tempfile.mkdtemp(prefix="clearvid-realesrgan-"))
    temp_video_path = temp_root / "enhanced_video.mp4"
    preview_path = temp_root / "preview.mp4"

    # Build a mux trigger that creates a playable preview with audio
    _preview_mux_lock = threading.Lock()
    _preview_mux_thread: list[threading.Thread | None] = [None]

    def _on_preview_mux_needed(frames_done: int) -> None:
        """Called from monitor loop when enough new frames warrant a preview update."""
        if preview_callback is None:
            return
        fps = metadata.fps or 30.0
        duration_sec = frames_done / fps
        if duration_sec < 10:  # not enough content yet
            return
        with _preview_mux_lock:
            if _preview_mux_thread[0] is not None and _preview_mux_thread[0].is_alive():
                return  # previous mux still running

            def _do_mux() -> None:
                ok = _mux_preview(config, temp_video_path, preview_path, duration_sec)
                if ok and preview_callback is not None:
                    preview_callback(str(preview_path))

            t = threading.Thread(target=_do_mux, daemon=True, name="preview-mux")
            _preview_mux_thread[0] = t
            t.start()

    try:
        outscale = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
        t_process = time.perf_counter()
        _stream_process_video(
            config=config,
            metadata=metadata,
            output_width=output_width,
            output_height=output_height,
            outscale=outscale,
            upsampler=upsampler,
            codeformer_restorer=codeformer_restorer,
            stabilizer=stabilizer,
            temp_video_path=temp_video_path,
            progress_callback=progress_callback,
            control=control,
            preview_mux_trigger=_on_preview_mux_needed,
        )
        t_process_elapsed = time.perf_counter() - t_process
        total_frames = _estimate_total_frames(metadata, config.preview_seconds) or 0
        avg_fps = total_frames / t_process_elapsed if t_process_elapsed > 0 and total_frames else 0
        _emit_progress(
            progress_callback, 95,
            f"帧处理完成: {total_frames}帧 {t_process_elapsed:.1f}s (平均 {avg_fps:.2f} fps)",
        )
        # Wait for any in-flight preview mux before final mux
        with _preview_mux_lock:
            if _preview_mux_thread[0] is not None:
                _preview_mux_thread[0].join(timeout=10)
        t_mux = time.perf_counter()
        _emit_progress(progress_callback, 96, "正在封装音频与元数据")
        _mux_output(config, temp_video_path)
        _emit_progress(
            progress_callback, 100,
            f"导出完成 | 总耗时 {time.perf_counter() - t_start:.1f}s "
            f"(初始化 {t_init:.1f}s + 帧处理 {t_process_elapsed:.1f}s + 封装 {time.perf_counter() - t_mux:.1f}s)",
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def enhance_single_frame(
    frame: np.ndarray,
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Enhance a single BGR frame using the full pipeline (upscale + face + sharpen).

    This is used by the preview feature to show a Before/After comparison
    without launching the full streaming pipeline.
    """
    weights_dir = REALESRGAN_WEIGHTS_DIR
    model_key = resolve_upscale_model(config.upscale_model, config.quality_mode)
    model_path = ensure_realesrgan_weights(weights_dir, model_key)
    upsampler = cast(Any, _build_upsampler(config, model_path, model_key, metadata.width, metadata.height))

    outscale = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
    enhanced, _ = upsampler.enhance(frame, outscale=outscale)

    face_restorer = _build_codeformer_restorer(config, metadata, output_width, output_height)
    if face_restorer is not None:
        enhanced = face_restorer.restore_faces(enhanced)

    enhanced = _resize_for_target(enhanced, output_width, output_height, config.target_profile)

    if config.sharpen_enabled and config.sharpen_strength > 0:
        enhanced = apply_sharpening(enhanced, config.sharpen_strength)

    return enhanced


def extract_frame(video_path: Path, timestamp_sec: float = 0.0, width: int = 0, height: int = 0) -> np.ndarray:
    return stream_codec.extract_frame(
        video_path,
        timestamp_sec=timestamp_sec,
        width=width,
        height=height,
        run_factory=subprocess.run,
    )
def _stream_process_video(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: Any,
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    temp_video_path: Path,
    progress_callback: Callable[[int, str], None] | None,
    control: ExportControl | None = None,
    preview_mux_trigger: Callable[[int], None] | None = None,
) -> None:
    decode_command = _build_decode_command(config, metadata)
    encode_command = _build_encode_command(config, metadata, output_width, output_height, temp_video_path)
    decoder, encoder = _start_stream_processes(decode_command, encode_command)

    try:
        _process_stream_frames(
            config=config,
            metadata=metadata,
            output_width=output_width,
            output_height=output_height,
            outscale=outscale,
            upsampler=upsampler,
            codeformer_restorer=codeformer_restorer,
            stabilizer=stabilizer,
            decoder=decoder,
            encoder=encoder,
            progress_callback=progress_callback,
            control=control,
            preview_mux_trigger=preview_mux_trigger,
        )
        _finalize_stream_processes(decoder, encoder)
    finally:
        _cleanup_stream_processes(decoder, encoder)


def _start_stream_processes(
    decode_command: list[str],
    encode_command: list[str],
) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]]:
    return stream_codec.start_stream_processes(
        decode_command,
        encode_command,
        popen_factory=subprocess.Popen,
    )
