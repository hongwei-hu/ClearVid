from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from clearvid.app.io.probe import collect_environment_info, probe_video
from clearvid.app.orchestrator import Orchestrator
from clearvid.app.pipeline import build_execution_plan
from clearvid.app.schemas.models import BackendType, EnhancementConfig, InferenceAccelerator, QualityMode, TargetProfile, UpscaleModel
from clearvid.app.task_queue import discover_video_files

app = typer.Typer(help="ClearVid command line interface")
console = Console()


@app.command()
def env() -> None:
    environment = collect_environment_info()
    table = Table(title="ClearVid Environment")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("ffmpeg", str(environment.ffmpeg_available))
    table.add_row("ffprobe", str(environment.ffprobe_available))
    table.add_row("nvidia-smi", str(environment.nvidia_smi_available))
    table.add_row("ffmpeg_version", environment.ffmpeg_version or "n/a")
    table.add_row("hwaccels", ", ".join(environment.ffmpeg_hwaccels) or "n/a")
    table.add_row("encoders", ", ".join(environment.ffmpeg_encoders) or "n/a")
    table.add_row("gpu_name", environment.gpu_name or "n/a")
    table.add_row("gpu_driver", environment.gpu_driver_version or "n/a")
    table.add_row("gpu_memory_mb", str(environment.gpu_memory_mb or "n/a"))
    table.add_row("torch_version", environment.torch_version or "n/a")
    table.add_row("torch_cuda", str(environment.torch_cuda_available))
    table.add_row("torch_gpu_compatible", str(environment.torch_gpu_compatible))
    table.add_row("realesrgan_available", str(environment.realesrgan_available))
    table.add_row("realesrgan_message", environment.realesrgan_message or "n/a")
    console.print(table)


@app.command()
def probe(input_path: Path) -> None:
    metadata = probe_video(input_path)
    console.print_json(data=json.loads(metadata.model_dump_json()))


@app.command()
def plan(
    input_path: Path,
    output_path: Path,
    target_profile: TargetProfile = typer.Option(TargetProfile.FHD),
    backend: BackendType = typer.Option(BackendType.AUTO),
    quality_mode: QualityMode = typer.Option(QualityMode.QUALITY),
) -> None:
    metadata = probe_video(input_path)
    config = EnhancementConfig(
        input_path=input_path,
        output_path=output_path,
        target_profile=target_profile,
        backend=backend,
        quality_mode=quality_mode,
        dry_run=True,
    )
    execution_plan = build_execution_plan(config, metadata)
    console.print_json(data=json.loads(execution_plan.model_dump_json()))


@app.command()
def run(
    input_path: Path,
    output_path: Path = typer.Option(..., "--output", help="Output video path"),
    target_profile: TargetProfile = typer.Option(TargetProfile.FHD),
    backend: BackendType = typer.Option(BackendType.AUTO),
    upscale_model: UpscaleModel = typer.Option(UpscaleModel.AUTO, help="Super-resolution model (auto/general_v3/x4plus)"),
    quality_mode: QualityMode = typer.Option(QualityMode.QUALITY),
    preserve_audio: bool = typer.Option(True),
    preserve_subtitles: bool = typer.Option(True),
    preserve_metadata: bool = typer.Option(True),
    face_restore_enabled: bool = typer.Option(True, help="Enable CodeFormer face restoration"),
    face_restore_strength: float = typer.Option(0.55, min=0.0, max=1.0, help="CodeFormer fidelity weight"),
    temporal_stabilize_enabled: bool = typer.Option(True, help="Enable optical-flow temporal stabilization"),
    temporal_stabilize_strength: float = typer.Option(0.6, min=0.0, max=1.0, help="Temporal stabilization strength"),
    preprocess_denoise: bool = typer.Option(True, help="Enable nlmeans denoise preprocessing"),
    preprocess_deblock: bool = typer.Option(True, help="Enable deblock preprocessing for low-bitrate H.264"),
    preprocess_deinterlace: str = typer.Option("auto", help="Deinterlace mode (auto/off)"),
    preprocess_colorspace_normalize: bool = typer.Option(True, help="Normalize colorspace to BT.709"),
    inference_accelerator: InferenceAccelerator = typer.Option(InferenceAccelerator.AUTO, help="Inference accelerator (none/auto/compile/tensorrt)"),
    async_pipeline: bool = typer.Option(True, "--async-pipeline/--no-async-pipeline", help="Enable 3-stage async pipeline"),
    preview_seconds: int | None = typer.Option(None, help="Process only the first N seconds"),
    dry_run: bool = typer.Option(False),
) -> None:
    config = EnhancementConfig(
        input_path=input_path,
        output_path=output_path,
        target_profile=target_profile,
        backend=backend,
        upscale_model=upscale_model,
        quality_mode=quality_mode,
        preserve_audio=preserve_audio,
        preserve_subtitles=preserve_subtitles,
        preserve_metadata=preserve_metadata,
        face_restore_enabled=face_restore_enabled,
        face_restore_strength=face_restore_strength,
        temporal_stabilize_enabled=temporal_stabilize_enabled,
        temporal_stabilize_strength=temporal_stabilize_strength,
        preprocess_denoise=preprocess_denoise,
        preprocess_deblock=preprocess_deblock,
        preprocess_deinterlace=preprocess_deinterlace,
        preprocess_colorspace_normalize=preprocess_colorspace_normalize,
        inference_accelerator=inference_accelerator,
        async_pipeline=async_pipeline,
        preview_seconds=preview_seconds,
        dry_run=dry_run,
    )
    result = Orchestrator().run_single(config)
    console.print(result.model_dump_json(indent=2))


@app.command()
def batch(
    input_dir: Path,
    output_dir: Path = typer.Option(..., "--output-dir", help="Batch output root"),
    target_profile: TargetProfile = typer.Option(TargetProfile.FHD),
    backend: BackendType = typer.Option(BackendType.AUTO),
    upscale_model: UpscaleModel = typer.Option(UpscaleModel.AUTO, help="Super-resolution model (auto/general_v3/x4plus)"),
    quality_mode: QualityMode = typer.Option(QualityMode.QUALITY),
    face_restore_enabled: bool = typer.Option(True, help="Enable CodeFormer face restoration"),
    face_restore_strength: float = typer.Option(0.55, min=0.0, max=1.0, help="CodeFormer fidelity weight"),
    temporal_stabilize_enabled: bool = typer.Option(True, help="Enable optical-flow temporal stabilization"),
    temporal_stabilize_strength: float = typer.Option(0.6, min=0.0, max=1.0, help="Temporal stabilization strength"),
    preprocess_denoise: bool = typer.Option(True, help="Enable nlmeans denoise preprocessing"),
    preprocess_deblock: bool = typer.Option(True, help="Enable deblock preprocessing for low-bitrate H.264"),
    preprocess_deinterlace: str = typer.Option("auto", help="Deinterlace mode (auto/off)"),
    preprocess_colorspace_normalize: bool = typer.Option(True, help="Normalize colorspace to BT.709"),
    inference_accelerator: InferenceAccelerator = typer.Option(InferenceAccelerator.AUTO, help="Inference accelerator (none/auto/compile/tensorrt)"),
    async_pipeline: bool = typer.Option(True, "--async-pipeline/--no-async-pipeline", help="Enable 3-stage async pipeline"),
    preview_seconds: int | None = typer.Option(None, help="Process only the first N seconds of each file"),
    dry_run: bool = typer.Option(False),
) -> None:
    files = discover_video_files(input_dir)
    if not files:
        raise typer.BadParameter(f"No supported video files found in {input_dir}")

    template = EnhancementConfig(
        input_path=files[0],
        output_path=output_dir / files[0].with_suffix(".mp4").name,
        target_profile=target_profile,
        backend=backend,
        upscale_model=upscale_model,
        quality_mode=quality_mode,
        face_restore_enabled=face_restore_enabled,
        face_restore_strength=face_restore_strength,
        temporal_stabilize_enabled=temporal_stabilize_enabled,
        temporal_stabilize_strength=temporal_stabilize_strength,
        preprocess_denoise=preprocess_denoise,
        preprocess_deblock=preprocess_deblock,
        preprocess_deinterlace=preprocess_deinterlace,
        preprocess_colorspace_normalize=preprocess_colorspace_normalize,
        inference_accelerator=inference_accelerator,
        async_pipeline=async_pipeline,
        preview_seconds=preview_seconds,
        dry_run=dry_run,
    )
    results = Orchestrator().run_batch(input_dir, output_dir, template)
    console.print_json(data=[json.loads(result.model_dump_json()) for result in results])


@app.command()
def gui() -> None:
    from clearvid.app.gui import main

    main()


if __name__ == "__main__":
    app()
