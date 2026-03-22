# ClearVid

ClearVid is a Windows-first video enhancement toolkit for real-world footage. It is designed around 480p and 720p inputs, with practical output targets such as 1080p, 4K, 2x scale, and 4x scale.

The project now includes a usable CLI, a desktop GUI, automatic environment/bootstrap logic, model weight management, and a modern Real-ESRGAN based processing path with optional face restoration and post-processing.

## Current status

What is implemented in the current repository state:

- Windows-first workflow for development and end-user launch.
- CLI commands for environment inspection, probing, planning, single-file runs, batch runs, and GUI startup.
- PySide6 desktop GUI with file management, preview, export queue, presets, history, environment diagnostics, and progress reporting.
- Smart recommendation flow for target profile, quality mode, model, encoder, accelerator, and tile suggestions.
- Automatic backend selection with fallback to baseline processing when model runtime is unavailable.
- Real-ESRGAN backend with two upscale model paths: `general_v3` and `x4plus`.
- Optional face restoration using either CodeFormer or GFPGAN.
- Preprocessing controls for denoise, deblock, deinterlace, and colorspace normalization.
- Post-processing controls for sharpening and optical-flow temporal stabilization.
- 8-bit and 10-bit export pixel formats including `yuv420p`, `yuv420p10le`, and `p010le`.
- Streaming raw-frame pipeline between FFmpeg and Python to reduce disk I/O without lowering export quality.
- Async multi-stage processing pipeline with dynamic batching and TensorRT engine cache support.
- First-run bootstrap flow for portable distributions, including dependency installation into `lib/`.
- Weight download management for Real-ESRGAN, CodeFormer, GFPGAN, and facelib dependencies.

Out of scope for the current version:

- Frame interpolation is not implemented.

## Recommended environment

- Windows 11
- Python 3.11 to 3.13
- FFmpeg and FFprobe available either in the project root or on `PATH`
- NVIDIA GPU with recent drivers for the Real-ESRGAN path
- CUDA-capable PyTorch environment for GPU inference

The repository `pyproject.toml` currently supports Python `>=3.11,<3.14`.

## Installation

### Source install

Create a virtual environment and install the base package:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

Install optional features as needed:

```powershell
python -m pip install -e .[gui,media]
python -m pip install -e .[inference]
python -m pip install -e .[acceleration]
python -m pip install -e .[dev]
```

Typical combinations:

- CLI only: base package
- GUI without AI inference: `.[gui,media]`
- Full local workstation setup: `.[gui,media,inference,acceleration]`

### Portable distribution build

The repository includes a distribution builder for a self-contained Windows package:

```powershell
.\scripts\build_dist.ps1
```

This creates a portable ZIP containing:

- Python embeddable runtime
- ClearVid source code
- launcher scripts
- FFmpeg binaries if found
- placeholder `lib/`, `outputs/`, `weights/`, and `samples/` directories

On first launch, the bootstrap launcher installs dependencies into `lib/` and then starts the GUI.

## Launching

### CLI entry points

After installation, the following entry points are available:

- `clearvid`
- `clearvid-gui`

### Windows launcher script

The repository also includes a double-click launcher:

```powershell
.\Start_ClearVid_GUI.bat
```

That script automatically detects whether it should start in:

- portable launcher mode, or
- development GUI mode

## CLI commands

Inspect the local environment:

```powershell
clearvid env
```

Probe a video file:

```powershell
clearvid probe samples\sample.mp4
```

Generate an execution plan and smart recommendation:

```powershell
clearvid plan samples\sample.mp4 outputs\sample_plan.mp4 --target-profile fhd --backend auto --quality-mode quality
```

Run a baseline validation export:

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend baseline --output outputs\sample_fhd_baseline.mp4
```

Run a quick preview pass on the first 20 seconds:

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend auto --preview-seconds 20 --output outputs\sample_preview.mp4
```

Run Real-ESRGAN with the higher-quality `x4plus` model, GFPGAN face restoration, TensorRT, and 10-bit output:

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend realesrgan --upscale-model x4plus --face-restore-model gfpgan --inference-accelerator tensorrt --output-pixel-format yuv420p10le --output outputs\sample_fhd_x4plus_10bit.mp4
```

Run a batch job for a folder:

```powershell
clearvid batch samples --target-profile fhd --backend auto --quality-mode balanced --output-dir outputs
```

Launch the GUI from the CLI:

```powershell
clearvid gui
```

Or directly:

```powershell
clearvid-gui
```

## Model and runtime behavior

### Available backends

- `auto`: prefer Real-ESRGAN when the runtime is available, otherwise fall back
- `realesrgan`: main quality path
- `baseline`: FFmpeg-based validation/fallback path

### Available upscale models

- `general_v3`: lighter default path
- `x4plus`: higher-quality RRDB path with heavier compute cost

### Available face restoration models

- `codeformer`: natural-looking restoration with fidelity control
- `gfpgan`: stronger beautification/restoration alternative

### Auto-downloaded weights

ClearVid can download the following weights on demand:

- `realesr-general-x4v3.pth`
- `RealESRGAN_x4plus.pth`
- `codeformer.pth`
- `GFPGANv1.4.pth`
- `detection_Resnet50_Final.pth`
- `parsing_parsenet.pth`

If you prefer to place weights manually, see [weights/README.zh-CN.md](weights/README.zh-CN.md).

## GUI highlights

The current GUI includes:

- three-column main window with file list, preview, and export panel
- built-in presets and smart parameter recommendations
- queue export and batch export
- live stage/progress updates during processing
- history dialog and recent-file tracking
- environment diagnostics and log panel
- output naming and export configuration import/export
- performance controls for accelerator selection, tile size, and batch size

## Performance notes

The current optimization strategy focuses on throughput without changing the final quality path.

Notable implemented optimizations:

- raw-frame streaming between FFmpeg and Python instead of PNG sequence round-trips
- async three-stage processing pipeline
- dynamic batch-size selection based on VRAM and model path
- dedicated TensorRT engine cache directory under `weights/trt_cache`
- GPU-first path with baseline fallback when inference runtime is not ready

The `fast` quality mode is intentionally more aggressive about disabling expensive post-processing to improve throughput.

## Project commands for contributors

Run tests:

```powershell
pytest
```

Run Ruff:

```powershell
ruff check .
```

## Notes

- This repository is Windows-first.
- FFmpeg must be reachable for both CLI and GUI workflows.
- The baseline backend remains useful as a diagnostic and fallback path, but it is not the primary quality target.
- The main quality path in the current codebase is Real-ESRGAN with optional face restoration and post-processing.
