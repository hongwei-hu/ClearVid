from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from clearvid.app.io.probe import collect_environment_info, probe_video
from clearvid.app.orchestrator import Orchestrator
from clearvid.app.pipeline import build_execution_plan
from clearvid.app.recommend import recommend
from clearvid.app.schemas.models import BackendType, EnhancementConfig, FaceRestoreModel, InferenceAccelerator, QualityMode, TargetProfile, UpscaleModel
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

    # Show smart recommendation
    environment = collect_environment_info()
    rec = recommend(metadata, environment)
    table = Table(title="智能推荐参数")
    table.add_column("参数")
    table.add_column("推荐值")
    table.add_row("输出规格", rec.target_profile)
    table.add_row("质量模式", rec.quality_mode)
    table.add_row("超分模型", rec.upscale_model)
    table.add_row("编码器", rec.encoder)
    table.add_row("推理加速", rec.inference_accelerator)
    table.add_row("分块尺寸", str(rec.tile_size) if rec.tile_size else "无分块")
    console.print(table)
    if rec.notes:
        for note in rec.notes:
            console.print(f"  [dim]•[/dim] {note}")


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
    face_restore_enabled: bool = typer.Option(True, help="Enable face restoration"),
    face_restore_strength: float = typer.Option(0.55, min=0.0, max=1.0, help="CodeFormer fidelity weight"),
    face_restore_model: FaceRestoreModel = typer.Option(FaceRestoreModel.CODEFORMER, help="Face restore model (codeformer/gfpgan)"),
    face_poisson_blend: bool = typer.Option(False, help="Use Poisson blending for face paste-back"),
    sharpen_enabled: bool = typer.Option(True, "--sharpen/--no-sharpen", help="Enable post-process sharpening"),
    sharpen_strength: float = typer.Option(0.12, min=0.0, max=1.0, help="Sharpening strength"),
    encoder_crf: int | None = typer.Option(None, help="Encoder CRF/CQ value (lower = higher quality)"),
    output_pixel_format: str = typer.Option("yuv420p", help="Output pixel format (yuv420p/yuv420p10le/p010le)"),
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
        face_restore_model=face_restore_model,
        face_poisson_blend=face_poisson_blend,
        sharpen_enabled=sharpen_enabled,
        sharpen_strength=sharpen_strength,
        encoder_crf=encoder_crf,
        output_pixel_format=output_pixel_format,
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
    face_restore_enabled: bool = typer.Option(True, help="Enable face restoration"),
    face_restore_strength: float = typer.Option(0.55, min=0.0, max=1.0, help="CodeFormer fidelity weight"),
    face_restore_model: FaceRestoreModel = typer.Option(FaceRestoreModel.CODEFORMER, help="Face restore model (codeformer/gfpgan)"),
    face_poisson_blend: bool = typer.Option(False, help="Use Poisson blending for face paste-back"),
    sharpen_enabled: bool = typer.Option(True, "--sharpen/--no-sharpen", help="Enable post-process sharpening"),
    sharpen_strength: float = typer.Option(0.12, min=0.0, max=1.0, help="Sharpening strength"),
    encoder_crf: int | None = typer.Option(None, help="Encoder CRF/CQ value"),
    output_pixel_format: str = typer.Option("yuv420p", help="Output pixel format"),
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
        face_restore_model=face_restore_model,
        face_poisson_blend=face_poisson_blend,
        sharpen_enabled=sharpen_enabled,
        sharpen_strength=sharpen_strength,
        encoder_crf=encoder_crf,
        output_pixel_format=output_pixel_format,
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
def warmup(
    model: UpscaleModel = typer.Option(UpscaleModel.GENERAL_V3, help="Model to pre-build engine for"),
    tile_size: int = typer.Option(512, help="Tile size for the optimization profile"),
    batch_size: int = typer.Option(1, help="Batch size"),
    fp16: bool = typer.Option(True, help="Use FP16 precision"),
    timeout: int | None = typer.Option(None, help="Build timeout override in seconds"),
    fast: bool = typer.Option(False, "--fast", help="Use standard (faster-build, higher-load) mode instead of low-load"),
) -> None:
    """Pre-build and cache the TensorRT engine for a model.

    Uses low-load mode by default (reduced GPU workspace, idle CPU priority)
    to keep the system responsive.  Pass --fast for standard mode if you
    prefer a shorter build time at the cost of system responsiveness.

    Run once after installing/updating ClearVid or changing GPU drivers.
    """
    import time as _time
    from pathlib import Path

    from clearvid.app.bootstrap.paths import REALESRGAN_WEIGHTS_DIR, TRT_CACHE_DIR
    from clearvid.app.models.realesrgan_runner import (
        _MODEL_REGISTRY,
        _build_upsampler,
        ensure_realesrgan_weights,
    )
    from clearvid.app.models.tensorrt_engine import (
        InferenceAccelerator,
        accelerate_model,
        check_engine_ready,
    )
    from clearvid.app.schemas.models import EnhancementConfig

    low_load = not fast
    model_key = model.value
    if model_key not in _MODEL_REGISTRY:
        console.print(f"[red]未知模型: {model_key}[/red]")
        raise typer.Exit(1)

    entry = _MODEL_REGISTRY[model_key]
    mode_label = "低负载" if low_load else "标准"
    console.print(f"[bold]预热模型: {entry['filename']} ({mode_label}模式)[/bold]")

    # Ensure weights
    weights_dir = REALESRGAN_WEIGHTS_DIR
    console.print("检查权重文件...")
    model_path = ensure_realesrgan_weights(weights_dir, model_key)
    console.print(f"  权重: {model_path}")

    # Build a minimal upsampler to get the model object
    dummy_config = EnhancementConfig(
        input_path=Path("warmup"),
        output_path=Path("warmup"),
        tile_size=tile_size,
        batch_size=batch_size,
        fp16_enabled=fp16,
    )
    upsampler = _build_upsampler(dummy_config, model_path, model_key, tile_size, tile_size)
    console.print(f"  架构: {entry['arch']}")

    # Check if already deployed
    ready, msg = check_engine_ready(
        upsampler.model,
        fp16=fp16, tile_size=tile_size, batch_size=batch_size,
        cache_dir=TRT_CACHE_DIR, weight_path=model_path,
    )
    if ready:
        console.print(f"\n[green]✓ {msg}[/green]")
        console.print(f"  缓存目录: {TRT_CACHE_DIR}")
        return

    # Build TRT engine
    console.print(f"\n[bold]构建 TensorRT 引擎 ({mode_label}模式)...[/bold]")
    console.print(f"  tile={tile_size}, batch={batch_size}, fp16={fp16}")
    if low_load:
        console.print(
            "[dim]  低负载模式: 构建时间较长，但系统可保持流畅操作[/dim]"
        )

    def _cb(pct: int, msg: str) -> None:
        console.print(f"  [{pct}%] {msg}")

    t0 = _time.perf_counter()
    try:
        accelerate_model(
            upsampler.model,
            InferenceAccelerator.TENSORRT,
            fp16=fp16,
            tile_size=tile_size,
            batch_size=batch_size,
            cache_dir=TRT_CACHE_DIR,
            progress_callback=_cb,
            weight_path=model_path,
            trt_build_timeout=timeout,
            build_if_missing=True,
            low_load=low_load,
        )
        elapsed = _time.perf_counter() - t0
        console.print(f"\n[green]✓ 引擎构建成功 (耗时 {elapsed:.1f}s)[/green]")
        console.print(f"  缓存目录: {TRT_CACHE_DIR}")
    except Exception as exc:
        console.print(f"\n[red]✗ 引擎构建失败: {exc}[/red]")
        console.print(
            "[dim]提示: 下次导出时将自动降级到 torch.compile 或标准 PyTorch。"
            " 删除 trt_cache/ 目录中的 .failed 文件可强制重试。[/dim]"
        )
        raise typer.Exit(1)


@app.command()
def gui() -> None:
    from clearvid.app.gui import main

    main()


if __name__ == "__main__":
    app()
