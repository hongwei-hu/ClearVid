# ClearVid

ClearVid is a Windows-first Python project for enhancing real-world character video, targeting 480p and 720p sources and outputting 1080p or 4K results.

## Current status

- Project skeleton is in place.
- FFmpeg and hardware capability probing are implemented.
- CLI and a simple desktop GUI are implemented.
- Batch job discovery and task planning are implemented.
- Baseline FFmpeg enhancement backend is implemented for end-to-end validation.
- Real-ESRGAN true backend is implemented and validated on the sample video.
- CodeFormer face restoration is integrated into the Real-ESRGAN pipeline.
- The GUI is directly launchable from Windows with the provided starter script.
- Real-ESRGAN runtime detection is implemented, with automatic fallback to the baseline backend when the model runtime is unavailable.

## Recommended environment

- Windows 11
- Python 3.11 or 3.12 recommended
- FFmpeg with NVENC and NVDEC support available on PATH
- NVIDIA GPU with recent drivers

The currently configured project `.venv` is using Python 3.13 on this machine and is already working with the installed CUDA-compatible PyTorch build.

## Install

Install the base toolchain:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e .[gui,media]
```

When you are ready to integrate model inference:

```powershell
python -m pip install -e .[inference]
```

## Commands

Probe a sample video:

```powershell
clearvid probe samples\sample.mp4
```

Inspect the local environment:

```powershell
clearvid env
```

Run the baseline enhancement backend:

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend baseline --output outputs\sample_fhd.mp4
```

Run a quick 20-second validation pass on a long sample:

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend baseline --preview-seconds 20 --output outputs\sample_fhd_preview.mp4
```

Batch a directory:

```powershell
clearvid batch samples --target-profile fhd --backend baseline --output-dir outputs
```

Run a quick Real-ESRGAN + CodeFormer preview:

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend realesrgan --preview-seconds 1 --output outputs\sample_fhd_realesrgan_codeformer_preview.mp4
```

Launch the GUI:

```powershell
clearvid gui
```

Or on Windows, just run:

```powershell
.\Start_ClearVid_GUI.bat
```

## Current model runtime note

The `.venv` environment in this repository is prepared for RTX 5090 compatible PyTorch, and the Real-ESRGAN true backend is available.

What is already ready:

- `.venv` contains a CUDA-compatible PyTorch build with `sm_120` support.
- ClearVid GUI and core dependencies are installed in `.venv`.
- Real-ESRGAN related Python packages are installed in `.venv`.
- ClearVid can auto-download the default `realesr-general-x4v3.pth` weight on first Real-ESRGAN run.
- ClearVid can auto-download `codeformer.pth` and required facelib detection/parsing weights on first face-restoration run.
- Auto backend selection now prefers Real-ESRGAN when the runtime is available.

Optional manual setup:

- You can still place custom Real-ESRGAN weights under [weights/README.zh-CN.md](weights/README.zh-CN.md)
- You can also pre-place CodeFormer and facelib weights under [weights/README.zh-CN.md](weights/README.zh-CN.md)
- The baseline backend remains available as a fallback path from both CLI and GUI

Run a quick Real-ESRGAN preview:

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend realesrgan --preview-seconds 2 --output outputs\sample_fhd_realesrgan_preview.mp4
```

## What the baseline backend does

The baseline backend is intentionally simple. It uses FFmpeg scaling and mild cleanup so the pipeline can be validated on a real sample before model integration.

It is not the final quality path.

The current final quality path is the Real-ESRGAN backend with optional CodeFormer face restoration. GUI and CLI both expose the face restoration switch and strength.
