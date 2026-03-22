# ClearVid

ClearVid is a Windows-first Python project for enhancing real-world character video, targeting 480p and 720p sources and outputting 1080p or 4K results.

## Current status

- Project skeleton is in place.
- FFmpeg and hardware capability probing are implemented.
- CLI and a simple desktop GUI are implemented.
- Batch job discovery and task planning are implemented.
- Baseline FFmpeg enhancement backend is implemented for end-to-end validation.
- The GUI is directly launchable from Windows with the provided starter script.
- Real-ESRGAN runtime detection is implemented, with automatic fallback to the baseline backend when the model runtime is unavailable.

## Recommended environment

- Windows 11
- Python 3.11 or 3.12 recommended
- FFmpeg with NVENC and NVDEC support available on PATH
- NVIDIA GPU with recent drivers

The currently configured local Python is 3.13. It is fine for the scaffold, but for heavy PyTorch and model compatibility, Python 3.11 is the safer target.

On this machine specifically, the installed PyTorch build does not support RTX 5090 GPU architecture yet, so the application currently auto-falls back to the FFmpeg baseline backend. The GUI will show this status clearly.

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
clearvid probe samples\480P_2000K_306841291pigmask.mp4
```

Inspect the local environment:

```powershell
clearvid env
```

Run the baseline enhancement backend:

```powershell
clearvid run samples\480P_2000K_306841291pigmask.mp4 --target-profile fhd --backend baseline --output outputs\sample_fhd.mp4
```

Run a quick 20-second validation pass on a long sample:

```powershell
clearvid run samples\480P_2000K_306841291pigmask.mp4 --target-profile fhd --backend baseline --preview-seconds 20 --output outputs\sample_fhd_preview.mp4
```

Batch a directory:

```powershell
clearvid batch samples --target-profile fhd --backend baseline --output-dir outputs
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

The `.venv` environment in this repository is now prepared for RTX 5090 compatible PyTorch.

What is already ready:

- `.venv` contains a CUDA-compatible PyTorch build with `sm_120` support.
- ClearVid GUI and core dependencies are installed in `.venv`.
- Real-ESRGAN related Python packages are installed in `.venv`.

What is still missing for the true model backend:

- Real-ESRGAN weight files under [weights/README.zh-CN.md](weights/README.zh-CN.md)

Until weights are added, the application will continue to run through the baseline backend, which is still directly usable from the GUI.

## What the baseline backend does

The baseline backend is intentionally simple. It uses FFmpeg scaling and mild cleanup so the pipeline can be validated on a real sample before model integration.

It is not the final quality path.

The next integration step is replacing the baseline video stage with Real-ESRGAN and attaching CodeFormer to the face enhancement stage.
