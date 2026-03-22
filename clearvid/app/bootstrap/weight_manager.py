"""Model weight manager — check existence and prompt for download.

Provides ``ensure_weight()`` which checks if a weight file exists and, when
used inside the GUI, shows a confirmation dialog before downloading.
The actual download is delegated to ``basicsr.utils.download_util`` which is
already used by the existing runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clearvid.app.bootstrap.paths import (
    CODEFORMER_WEIGHTS_DIR,
    FACELIB_WEIGHTS_DIR,
    GFPGAN_WEIGHTS_DIR,
    REALESRGAN_WEIGHTS_DIR,
)


@dataclass(frozen=True)
class WeightSpec:
    """Describes a single model weight file."""

    name: str  # human-readable
    filename: str
    directory: Path
    url: str
    size_mb: int  # approximate size for the download prompt

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    @property
    def exists(self) -> bool:
        return self.path.is_file()


# ---------------------------------------------------------------------------
# Registry of known weights
# ---------------------------------------------------------------------------

WEIGHT_REGISTRY: dict[str, WeightSpec] = {
    "realesrgan_general_v3": WeightSpec(
        name="Real-ESRGAN General v3",
        filename="realesr-general-x4v3.pth",
        directory=REALESRGAN_WEIGHTS_DIR,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        size_mb=65,
    ),
    "realesrgan_x4plus": WeightSpec(
        name="Real-ESRGAN x4plus",
        filename="RealESRGAN_x4plus.pth",
        directory=REALESRGAN_WEIGHTS_DIR,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        size_mb=64,
    ),
    "codeformer": WeightSpec(
        name="CodeFormer 人脸修复",
        filename="codeformer.pth",
        directory=CODEFORMER_WEIGHTS_DIR,
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        size_mb=376,
    ),
    "gfpgan_v1.4": WeightSpec(
        name="GFPGAN v1.4 人脸修复",
        filename="GFPGANv1.4.pth",
        directory=GFPGAN_WEIGHTS_DIR,
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        size_mb=332,
    ),
    "facelib_detection": WeightSpec(
        name="人脸检测模型",
        filename="detection_Resnet50_Final.pth",
        directory=FACELIB_WEIGHTS_DIR,
        url="https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        size_mb=104,
    ),
    "facelib_parsing": WeightSpec(
        name="人脸解析模型",
        filename="parsing_parsenet.pth",
        directory=FACELIB_WEIGHTS_DIR,
        url="https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        size_mb=82,
    ),
}


def missing_weights_for_export(
    *,
    face_restore_enabled: bool = False,
    face_restore_model: str = "codeformer",
    upscale_model: str = "auto",
) -> list[WeightSpec]:
    """Return a list of WeightSpec that are needed but not yet downloaded."""
    needed: list[WeightSpec] = []

    # Real-ESRGAN model is always needed
    if upscale_model in ("x4plus",):
        key = "realesrgan_x4plus"
    else:
        key = "realesrgan_general_v3"
    spec = WEIGHT_REGISTRY[key]
    if not spec.exists:
        needed.append(spec)

    # Face restoration models
    if face_restore_enabled:
        if face_restore_model == "gfpgan":
            face_key = "gfpgan_v1.4"
        else:
            face_key = "codeformer"
        face_spec = WEIGHT_REGISTRY[face_key]
        if not face_spec.exists:
            needed.append(face_spec)

        # Face detection + parsing always needed with face restore
        for aux_key in ("facelib_detection", "facelib_parsing"):
            aux_spec = WEIGHT_REGISTRY[aux_key]
            if not aux_spec.exists:
                needed.append(aux_spec)

    return needed


def download_weight(spec: WeightSpec, on_progress: callable | None = None) -> bool:
    """Download a single weight file.  Returns True on success.

    Falls back to urllib if basicsr download utility is unavailable.
    """
    spec.directory.mkdir(parents=True, exist_ok=True)

    # Try basicsr download (shows progress bar in console automatically)
    try:
        from basicsr.utils.download_util import load_file_from_url

        load_file_from_url(
            url=spec.url,
            model_dir=str(spec.directory),
            file_name=spec.filename,
        )
        return spec.exists
    except ImportError:
        pass

    # Fallback to urllib with manual progress
    import urllib.request

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if on_progress and total_size > 0:
            on_progress(min(100, int(block_num * block_size * 100 / total_size)))

    try:
        urllib.request.urlretrieve(spec.url, str(spec.path), reporthook=_report)  # noqa: S310
        return spec.exists
    except Exception:  # noqa: BLE001
        return False


def format_download_prompt(missing: list[WeightSpec]) -> str:
    """Return a user-friendly string describing what needs to be downloaded."""
    total_mb = sum(s.size_mb for s in missing)
    lines = [f"需要下载以下模型权重 (共约 {total_mb} MB):"]
    for s in missing:
        lines.append(f"  • {s.name} ({s.size_mb} MB)")
    lines.append("\n是否继续？")
    return "\n".join(lines)
