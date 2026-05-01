"""Export estimation: rough time and file-size predictions.

Accounts for the actual model type and accelerator state rather than using
baked-in per-mode FPS constants.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportEstimate:
    """Rough export estimation."""

    estimated_seconds: float
    estimated_size_mb: float
    description: str


# Base processing speed in frames-per-second by model on a mid-range GPU
# (pure PyTorch, fp16, tile=512).  Actual throughput varies with tile size,
# resolution, and batch size.
_BASE_FPS_BY_MODEL: dict[str, float] = {
    "general_v3": 15.0,   # SRVGGNetCompact — lightweight
    "x4plus":      4.0,   # RRDBNet — heavy, 23 residual blocks
}

# Multiplier per accelerator type
_ACCEL_MULTIPLIER: dict[str, float] = {
    "tensorrt":      3.0,   # 2-4x speedup
    "torch_compile": 1.6,   # 1.3-1.8x
    "pytorch":       1.0,   # baseline
}

# Post-processing overhead per enabled feature (subtractive multipliers)
_PP_OVERHEAD: dict[str, float] = {
    "face_restore":  0.55,   # reduces throughput by ~45%
    "temporal":      0.85,   # reduces throughput by ~15%
    "sharpen":       0.97,   # nearly free
}


# CRF -> approximate bitrate multiplier relative to source
# Lower CRF = higher quality = larger file
_CRF_SIZE_MULT: dict[int, float] = {
    0: 4.0,    # near lossless
    10: 2.5,
    15: 1.8,
    18: 1.2,
    22: 0.8,
    28: 0.5,
    35: 0.3,
    51: 0.15,
}

# Resolution scale factor on file size (relative to 1080p baseline)
_RES_SIZE_MULT: dict[str, float] = {
    "source": 1.0,
    "fhd": 1.0,
    "uhd4k": 3.5,
    "scale2x": 3.0,
    "scale4x": 10.0,
}


def estimate_export(
    duration_sec: float,
    total_frames: int,
    *,
    quality_mode: str = "quality",
    target_profile: str = "fhd",
    encoder_crf: int | None = 18,
    source_size_bytes: int = 0,
    upscale_model: str = "general_v3",
    accelerator: str = "pytorch",
    face_restore: bool = True,
    temporal_stabilize: bool = True,
    sharpen: bool = True,
) -> ExportEstimate:
    """Produce a rough export estimate.

    Parameters
    ----------
    duration_sec:
        Video duration in seconds.
    total_frames:
        Total number of frames.
    quality_mode:
        One of fast / balanced / quality. FAST mode disables post-processing.
    target_profile:
        One of source / fhd / uhd4k / scale2x / scale4x.
    encoder_crf:
        CRF value (0-51). None defaults to 18.
    source_size_bytes:
        Original file size. Used for size estimation.
    upscale_model:
        Model key (general_v3 / x4plus). Affects base FPS.
    accelerator:
        Active accelerator (tensorrt / torch_compile / pytorch).
    face_restore:
        Whether face restoration is enabled.
    temporal_stabilize:
        Whether temporal stabilization is enabled.
    sharpen:
        Whether sharpening is enabled.
    """

    if total_frames <= 0:
        total_frames = max(1, int(duration_sec * 30))

    # --- Time estimate -------------------------------------------------------
    # Base FPS from model
    base_fps = _BASE_FPS_BY_MODEL.get(upscale_model, 15.0)

    # Accelerator multiplier
    accel_mult = _ACCEL_MULTIPLIER.get(accelerator, 1.0)

    # Post-processing overhead (FAST mode skips all postprocessing)
    pp_mult = 1.0
    if quality_mode != "fast":
        if face_restore:
            pp_mult *= _PP_OVERHEAD["face_restore"]
        if temporal_stabilize:
            pp_mult *= _PP_OVERHEAD["temporal"]
        if sharpen:
            pp_mult *= _PP_OVERHEAD["sharpen"]

    fps = base_fps * accel_mult * pp_mult
    est_seconds = total_frames / max(fps, 0.5)  # floor at 0.5 FPS

    # --- Size estimate -------------------------------------------------------
    crf = encoder_crf if encoder_crf is not None else 18
    crf_mult = _interpolate_crf_mult(crf)
    res_mult = _RES_SIZE_MULT.get(target_profile, 1.0)

    if source_size_bytes > 0:
        est_bytes = source_size_bytes * crf_mult * res_mult
    else:
        est_bytes = duration_sec * 2.0 * 1024 * 1024 * crf_mult * res_mult

    est_mb = est_bytes / (1024 * 1024)

    # --- Description ---------------------------------------------------------
    time_str = format_duration(est_seconds)
    size_str = f"{est_mb:.0f} MB" if est_mb < 1024 else f"{est_mb / 1024:.1f} GB"

    accel_note = ""
    if accelerator == "tensorrt":
        accel_note = " (TensorRT)"
    elif accelerator == "torch_compile":
        accel_note = " (torch.compile)"
    model_note = "x4plus" if upscale_model == "x4plus" else ""
    pp_note = " 快速模式" if quality_mode == "fast" else ""
    desc_parts = [f"预计耗时 {time_str}", f"输出约 {size_str}"]
    if accel_note or model_note or pp_note:
        detail = " ".join(p for p in [accel_note, model_note, pp_note] if p)
        desc_parts.append(f"({detail.strip()})")
    desc = "，".join(desc_parts)

    return ExportEstimate(
        estimated_seconds=est_seconds,
        estimated_size_mb=est_mb,
        description=desc,
    )


def _interpolate_crf_mult(crf: int) -> float:
    """Linear interpolation from the CRF -> multiplier table."""
    keys = sorted(_CRF_SIZE_MULT.keys())
    if crf <= keys[0]:
        return _CRF_SIZE_MULT[keys[0]]
    if crf >= keys[-1]:
        return _CRF_SIZE_MULT[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= crf <= hi:
            frac = (crf - lo) / (hi - lo)
            return _CRF_SIZE_MULT[lo] + frac * (_CRF_SIZE_MULT[hi] - _CRF_SIZE_MULT[lo])
    return 1.0


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m} 分 {s} 秒"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h} 小时 {m} 分"
