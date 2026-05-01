"""Background worker threads for video processing, preview, and TRT deployment."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from clearvid.app.bootstrap.paths import REALESRGAN_WEIGHTS_DIR, TRT_CACHE_DIR
from clearvid.app.export_control import ExportCancelled, ExportControl
from clearvid.app.models.realesrgan_runner import (
    _MODEL_REGISTRY,
    _build_upsampler,
    ensure_realesrgan_weights,
    resolve_upscale_model,
)
from clearvid.app.models.tensorrt_engine import (
    InferenceAccelerator,
    accelerate_model,
    check_engine_ready,
)
from clearvid.app.orchestrator import Orchestrator
from clearvid.app.schemas.models import EnhancementConfig, QualityMode, UpscaleModel


class Worker(QThread):
    """Background thread for single-file video enhancement."""

    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int, str)
    preview_ready = Signal(str)

    def __init__(self, config: EnhancementConfig, control: ExportControl | None = None) -> None:
        super().__init__()
        self._config = config
        self._control = control

    def run(self) -> None:
        try:
            result = Orchestrator().run_single(
                self._config, progress_callback=self._emit_progress,
                control=self._control,
                preview_callback=self._emit_preview,
            )
            self.completed.emit(result.model_dump_json(indent=2))
        except ExportCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _emit_progress(self, percent: int, message: str) -> None:
        self.progress.emit(percent, message)

    def _emit_preview(self, path: str) -> None:
        self.preview_ready.emit(path)


class PreviewWorker(QThread):
    """Background thread for single-frame preview generation."""

    finished = Signal(object, object)  # (original_bgr, enhanced_bgr)
    failed = Signal(str)

    def __init__(self, config: EnhancementConfig, timestamp_sec: float) -> None:
        super().__init__()
        self._config = config
        self._timestamp_sec = timestamp_sec

    def run(self) -> None:
        try:
            original, enhanced, _ = Orchestrator().preview_frame(
                self._config, self._timestamp_sec
            )
            self.finished.emit(original, enhanced)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class TrtWarmupWorker(QThread):
    """Background thread for TensorRT engine deployment.

    Builds the engine in a subprocess with low-load settings so the GUI stays
    responsive.  Reports progress via signals.
    """

    ready = Signal(str)         # engine_path
    failed = Signal(str)        # error message
    progress = Signal(int, str)  # percent 0-100, message
    done = Signal()             # completed (success or failure)

    def __init__(
        self,
        model_key: str = "general_v3",
        tile_size: int = 512,
        batch_size: int = 1,
        fp16: bool = True,
        timeout: int | None = None,
    ) -> None:
        super().__init__()
        self._model_key = model_key
        self._tile_size = tile_size
        self._batch_size = batch_size
        self._fp16 = fp16
        self._timeout = timeout

    def run(self) -> None:
        try:
            weights_dir = REALESRGAN_WEIGHTS_DIR
            model_key = self._model_key
            if model_key not in _MODEL_REGISTRY:
                self.failed.emit(f"未知模型: {model_key}")
                return

            entry = _MODEL_REGISTRY[model_key]
            self.progress.emit(5, f"准备权重: {entry['filename']}")

            # Ensure weights exist
            model_path = ensure_realesrgan_weights(weights_dir, model_key)

            # Build a minimal upsampler to get the model object
            dummy_config = EnhancementConfig(
                input_path=Path("warmup"),
                output_path=Path("warmup"),
                tile_size=self._tile_size,
                batch_size=self._batch_size,
                fp16_enabled=self._fp16,
            )
            self.progress.emit(10, f"加载模型架构: {entry['arch']}")
            upsampler = _build_upsampler(
                dummy_config, model_path, model_key,
                self._tile_size, self._tile_size,
            )

            # Check if already deployed
            ready, msg = check_engine_ready(
                upsampler.model,
                fp16=self._fp16,
                tile_size=self._tile_size,
                batch_size=self._batch_size,
                cache_dir=TRT_CACHE_DIR,
                weight_path=model_path,
            )
            if ready:
                self.progress.emit(100, "TensorRT 引擎已就绪")
                return  # done emitted by finally

            self.progress.emit(15, msg)

            # Build TRT engine
            def _cb(pct: int, msg: str) -> None:
                # Map 0-100 from builder to our progress range (15-95)
                mapped = 15 + int(pct * 0.8)
                self.progress.emit(mapped, msg)

            try:
                accelerate_model(
                    upsampler.model,
                    InferenceAccelerator.TENSORRT,
                    fp16=self._fp16,
                    tile_size=self._tile_size,
                    batch_size=self._batch_size,
                    cache_dir=TRT_CACHE_DIR,
                    progress_callback=_cb,
                    weight_path=model_path,
                    trt_build_timeout=self._timeout,
                    build_if_missing=True,
                    low_load=True,
                )
                self.progress.emit(100, "TensorRT 引擎部署完成")
            except Exception as exc:
                self.failed.emit(f"{exc}\n\n--- 完整异常信息 ---\n{traceback.format_exc()}")  # noqa: BLE001
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n\n--- 完整异常信息 ---\n{traceback.format_exc()}")
        finally:
            self.done.emit()
