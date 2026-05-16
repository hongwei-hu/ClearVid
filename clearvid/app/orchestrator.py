from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from clearvid.app.export_control import ExportControl
from clearvid.app.io.probe import probe_video
from clearvid.app.models.realesrgan_runner import (
    enhance_single_frame,
    extract_frame,
    run_realesrgan_video,
)
from clearvid.app.pipeline import build_execution_plan, resolve_output_size
from clearvid.app.schemas.models import BatchResult, EnhancementConfig
from clearvid.app.task_queue import discover_video_files
from clearvid.app.utils.subprocess_utils import run_ffmpeg_with_progress


class Orchestrator:
    def preview_frame(
        self,
        config: EnhancementConfig,
        timestamp_sec: float = 0.0,
        control: ExportControl | None = None,
    ) -> tuple:
        """Extract and enhance a single frame for Before/After preview.

        Returns ``(original_bgr, enhanced_bgr, metadata)`` numpy arrays.
        """

        if control is not None:
            control.check()
        metadata = probe_video(config.input_path)
        if control is not None:
            control.check()
        original = extract_frame(config.input_path, timestamp_sec, width=metadata.width, height=metadata.height)
        if control is not None:
            control.check()
        output_width, output_height = resolve_output_size(
            metadata.width, metadata.height, config.target_profile,
        )
        enhanced = enhance_single_frame(original, config, metadata, output_width, output_height)
        if control is not None:
            control.check()
        return original, enhanced, metadata

    def run_single(
        self,
        config: EnhancementConfig,
        progress_callback: Callable[[int, str], None] | None = None,
        control: ExportControl | None = None,
        preview_callback: Callable[[str], None] | None = None,
    ) -> BatchResult:
        _emit_progress(progress_callback, 2, "正在分析输入视频")
        metadata = probe_video(config.input_path)
        plan = build_execution_plan(config, metadata)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.dry_run:
            return BatchResult(
                input_path=config.input_path,
                output_path=config.output_path,
                success=True,
                message="Dry run created execution plan only.",
                backend=plan.backend,
            )

        if plan.backend.value == "realesrgan":
            _emit_progress(progress_callback, 5, "正在加载 Real-ESRGAN 与 CodeFormer")
            run_realesrgan_video(
                config=config,
                metadata=metadata,
                output_width=plan.output_width,
                output_height=plan.output_height,
                progress_callback=progress_callback,
                control=control,
                preview_callback=preview_callback,
            )
        else:
            run_ffmpeg_with_progress(
                plan.command,
                duration_seconds=config.preview_seconds or metadata.duration_seconds,
                progress_callback=progress_callback,
                progress_message="正在执行基线增强",
                progress_start=5,
                progress_end=100,
            )

        _emit_progress(progress_callback, 100, "导出完成")
        return BatchResult(
            input_path=config.input_path,
            output_path=config.output_path,
            success=True,
            message=f"Finished with backend {plan.backend.value}.",
            backend=plan.backend,
        )

    def run_batch(self, input_dir: Path, output_dir: Path, template: EnhancementConfig) -> list[BatchResult]:
        results: list[BatchResult] = []
        for input_path in discover_video_files(input_dir):
            relative = input_path.relative_to(input_dir)
            output_relative = relative.parent / f"{relative.stem}_{template.target_profile.value}.mp4"
            output_path = output_dir / output_relative
            config = template.model_copy(update={"input_path": input_path, "output_path": output_path})

            try:
                results.append(self.run_single(config))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    BatchResult(
                        input_path=input_path,
                        output_path=output_path,
                        success=False,
                        message=str(exc),
                        backend=None,
                    )
                )

        return results


def _emit_progress(
    progress_callback: Callable[[int, str], None] | None,
    percent: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)
