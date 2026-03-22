from __future__ import annotations

from pathlib import Path

from clearvid.app.io.probe import probe_video
from clearvid.app.pipeline import build_execution_plan
from clearvid.app.schemas.models import BatchResult, EnhancementConfig
from clearvid.app.task_queue import discover_video_files
from clearvid.app.utils.subprocess_utils import run_command


class Orchestrator:
    def run_single(self, config: EnhancementConfig) -> BatchResult:
        metadata = probe_video(config.input_path)
        plan = build_execution_plan(config, metadata)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.dry_run:
            return BatchResult(
                input_path=config.input_path,
                output_path=config.output_path,
                success=True,
                message="Dry run created execution plan only.",
            )

        run_command(plan.command)
        return BatchResult(
            input_path=config.input_path,
            output_path=config.output_path,
            success=True,
            message=f"Finished with backend {config.backend.value}.",
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
                    )
                )

        return results
