"""Export estimation: rough time and file-size predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportEstimate:
    """Rough export estimation."""

    estimated_seconds: float
    estimated_size_mb: float
    description: str


# Rough processing speed in frames-per-second by quality mode (GPU).
# These are conservative estimates for a mid-range card (RTX 3070-class).
_FPS_BY_QUALITY: dict[str, float] = {
    "fast": 8.0,
    "balanced": 4.0,
    "quality": 2.0,
}

# CRF → approximate bitrate multiplier relative to source
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
    quality_mode: str = "quality",
    target_profile: str = "fhd",
    encoder_crf: int | None = 18,
    source_size_bytes: int = 0,
) -> ExportEstimate:
    """Produce a rough export estimate.

    Parameters
    ----------
    duration_sec : float
        Video duration in seconds.
    total_frames : int
        Total number of frames.
    quality_mode : str
        One of fast / balanced / quality.
    target_profile : str
        One of source / fhd / uhd4k / scale2x / scale4x.
    encoder_crf : int or None
        CRF value (0-51). None defaults to 18.
    source_size_bytes : int
        Original file size. Used for size estimation.

    Returns
    -------
    ExportEstimate
    """
    if total_frames <= 0:
        total_frames = max(1, int(duration_sec * 30))

    # --- Time estimate ---
    fps = _FPS_BY_QUALITY.get(quality_mode, 2.0)
    est_seconds = total_frames / fps

    # --- Size estimate ---
    crf = encoder_crf if encoder_crf is not None else 18
    # Interpolate CRF multiplier
    crf_mult = _interpolate_crf_mult(crf)
    res_mult = _RES_SIZE_MULT.get(target_profile, 1.0)

    if source_size_bytes > 0:
        est_bytes = source_size_bytes * crf_mult * res_mult
    else:
        # Fallback: ~2 MB per second of 1080p at CRF 18
        est_bytes = duration_sec * 2.0 * 1024 * 1024 * crf_mult * res_mult

    est_mb = est_bytes / (1024 * 1024)

    # --- Description ---
    time_str = format_duration(est_seconds)
    size_str = f"{est_mb:.0f} MB" if est_mb < 1024 else f"{est_mb / 1024:.1f} GB"
    desc = f"预计耗时 {time_str}，输出约 {size_str}"

    return ExportEstimate(
        estimated_seconds=est_seconds,
        estimated_size_mb=est_mb,
        description=desc,
    )


def _interpolate_crf_mult(crf: int) -> float:
    """Linear interpolation from the CRF → multiplier table."""
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
