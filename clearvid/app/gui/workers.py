"""Background worker threads for video processing, preview, and TRT deployment."""

from __future__ import annotations

import logging
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
    trt_profile_fallbacks,
)
from clearvid.app.orchestrator import Orchestrator
from clearvid.app.schemas.models import EnhancementConfig, QualityMode, UpscaleModel

logger = logging.getLogger(__name__)


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
        low_load: bool = False,
        allow_fallbacks: bool = True,
    ) -> None:
        super().__init__()
        self._model_key = model_key
        self._tile_size = tile_size
        self._batch_size = batch_size
        self._fp16 = fp16
        self._timeout = timeout
        self._low_load = low_load
        self._allow_fallbacks = allow_fallbacks

    def run(self) -> None:
        try:
            self._log_warmup_started()
            weights_dir = REALESRGAN_WEIGHTS_DIR
            model_key = self._model_key
            if model_key not in _MODEL_REGISTRY:
                logger.error("TRT warmup aborted: unknown model %s", model_key)
                self.failed.emit(f"未知模型: {model_key}")
                return

            entry = _MODEL_REGISTRY[model_key]
            self.progress.emit(5, f"准备权重: {entry['filename']}")

            # Ensure weights exist
            model_path = ensure_realesrgan_weights(weights_dir, model_key)

            errors: list[str] = []
            profiles = self._candidate_profiles()
            for profile_index, (tile_size, batch_size) in enumerate(profiles, start=1):
                should_stop = self._try_profile(
                    model_key, entry, model_path, profile_index, len(profiles),
                    tile_size, batch_size, errors,
                )
                if should_stop:
                    return
        except Exception as exc:  # noqa: BLE001
            logger.exception("TRT warmup crashed")
            self.failed.emit(f"{exc}\n\n--- 完整异常信息 ---\n{traceback.format_exc()}")
        finally:
            logger.info("TRT warmup finished: model=%s", self._model_key)
            self.done.emit()

    def _log_warmup_started(self) -> None:
        logger.info(
            "TRT warmup started: model=%s tile=%d batch=%d fp16=%s low_load=%s fallback=%s",
            self._model_key,
            self._tile_size,
            self._batch_size,
            self._fp16,
            self._low_load,
            self._allow_fallbacks,
        )

    def _candidate_profiles(self) -> list[tuple[int, int]]:
        if self._allow_fallbacks:
            return trt_profile_fallbacks(self._tile_size, self._batch_size)
        return [(self._tile_size, self._batch_size)]

    def _try_profile(
        self,
        model_key: str,
        entry: dict,
        model_path: Path,
        profile_index: int,
        profile_count: int,
        tile_size: int,
        batch_size: int,
        errors: list[str],
    ) -> bool:
        try:
            self._log_profile_attempt(model_key, profile_index, profile_count, tile_size, batch_size)
            upsampler = self._build_warmup_upsampler(entry, model_path, model_key, tile_size, batch_size)
            ready, msg = self._check_profile_status(upsampler.model, model_path, tile_size, batch_size)
            if ready:
                logger.info(
                    "TRT engine already ready: model=%s tile=%d batch=%d",
                    model_key, tile_size, batch_size,
                )
                self.progress.emit(100, f"TensorRT 引擎已就绪 (tile={tile_size}, batch={batch_size})")
                return True

            self.progress.emit(15, msg)
            if self._should_skip_fallback_profile(profile_index, msg):
                return self._skip_failed_fallback(
                    model_key, tile_size, batch_size, msg, errors,
                    profile_index, profile_count,
                )

            self._deploy_profile(upsampler.model, model_path, tile_size, batch_size)
            logger.info(
                "TRT engine deployed: model=%s tile=%d batch=%d",
                model_key, tile_size, batch_size,
            )
            self.progress.emit(100, f"TensorRT 引擎部署完成 (tile={tile_size}, batch={batch_size})")
            return True
        except Exception as exc:  # noqa: BLE001
            return self._handle_profile_exception(
                exc, model_key, tile_size, batch_size,
                profile_index, profile_count, errors,
            )

    def _log_profile_attempt(
        self,
        model_key: str,
        profile_index: int,
        profile_count: int,
        tile_size: int,
        batch_size: int,
    ) -> None:
        logger.info(
            "TRT warmup profile attempt %d/%d: model=%s tile=%d batch=%d",
            profile_index,
            profile_count,
            model_key,
            tile_size,
            batch_size,
        )

    def _build_warmup_upsampler(
        self,
        entry: dict,
        model_path: Path,
        model_key: str,
        tile_size: int,
        batch_size: int,
    ):
        self.progress.emit(
            10,
            f"加载模型架构: {entry['arch']} (tile={tile_size}, batch={batch_size})",
        )
        dummy_config = EnhancementConfig(
            input_path=Path("warmup"),
            output_path=Path("warmup"),
            tile_size=tile_size,
            batch_size=batch_size,
            fp16_enabled=self._fp16,
        )
        return _build_upsampler(
            dummy_config, model_path, model_key,
            tile_size, tile_size,
        )

    def _check_profile_status(
        self,
        model,
        model_path: Path,
        tile_size: int,
        batch_size: int,
    ) -> tuple[bool, str]:
        return check_engine_ready(
            model,
            fp16=self._fp16,
            tile_size=tile_size,
            batch_size=batch_size,
            cache_dir=TRT_CACHE_DIR,
            weight_path=model_path,
        )

    def _deploy_profile(
        self,
        model,
        model_path: Path,
        tile_size: int,
        batch_size: int,
    ) -> None:
        def _cb(pct: int, msg: str) -> None:
            mapped = 15 + int(pct * 0.8)
            self.progress.emit(mapped, msg)

        accelerate_model(
            model,
            InferenceAccelerator.TENSORRT,
            fp16=self._fp16,
            tile_size=tile_size,
            batch_size=batch_size,
            cache_dir=TRT_CACHE_DIR,
            progress_callback=_cb,
            weight_path=model_path,
            trt_build_timeout=self._timeout,
            build_if_missing=True,
            low_load=self._low_load,
        )

    def _should_skip_fallback_profile(self, profile_index: int, message: str) -> bool:
        return (
            self._allow_fallbacks
            and profile_index > 1
            and self._is_recent_failed_status(message)
        )

    def _skip_failed_fallback(
        self,
        model_key: str,
        tile_size: int,
        batch_size: int,
        message: str,
        errors: list[str],
        profile_index: int,
        profile_count: int,
    ) -> bool:
        reason = f"跳过近期失败记录 ({message})"
        logger.info(
            "Skipping recently failed TRT fallback profile: model=%s tile=%d batch=%d status=%s",
            model_key, tile_size, batch_size, message,
        )
        errors.append(f"tile={tile_size}, batch={batch_size}: {reason}")
        if profile_index < profile_count:
            self.progress.emit(15, f"{reason}，继续尝试下一个组合")
            return False
        self.failed.emit("TensorRT 引擎部署失败，已尝试以下配置:\n" + "\n".join(errors))
        return True

    def _handle_profile_exception(
        self,
        exc: Exception,
        model_key: str,
        tile_size: int,
        batch_size: int,
        profile_index: int,
        profile_count: int,
        errors: list[str],
    ) -> bool:
        reason = self._summarize_trt_failure(exc)
        errors.append(f"tile={tile_size}, batch={batch_size}: {reason}")
        if profile_index < profile_count:
            logger.warning(
                "TRT warmup fallback profile failed: model=%s tile=%d batch=%d reason=%s",
                model_key, tile_size, batch_size, reason,
            )
            self.progress.emit(15, f"当前配置失败 ({reason})，继续尝试下一个组合")
            self._clear_cuda_cache()
            return False
        logger.exception(
            "TRT warmup final profile failed: model=%s tile=%d batch=%d reason=%s",
            model_key, tile_size, batch_size, reason,
        )
        self.failed.emit(
            "TensorRT 引擎部署失败，已尝试以下配置:\n"
            + "\n".join(errors)
            + f"\n\n--- 完整异常信息 ---\n{traceback.format_exc()}"
        )
        return True

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _is_recent_failed_status(message: str) -> bool:
        return message.startswith("上次部署失败")

    @staticmethod
    def _summarize_trt_failure(exc: Exception) -> str:
        text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        if "ENGINE_BUILD_RETURNED_NONE_ALL_MODES" in str(exc):
            return "TensorRT builder 在当前 profile 的所有构建模式下均未生成 engine"
        if "ENGINE_BUILD_RETURNED_NONE" in str(exc):
            return "TensorRT builder 未能为该 profile 生成 engine"
        if "CUDA out of memory" in str(exc) or "OutOfMemoryError" in str(exc):
            return "CUDA 显存不足"
        if "timed out" in str(exc).lower() or "超时" in str(exc):
            return "构建超时"
        return text[:160]
