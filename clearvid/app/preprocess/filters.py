"""FFmpeg video-filter builders for the preprocessing chain.

Each public function returns a list of FFmpeg ``-vf`` filter strings (may be
empty when the filter is disabled).  The caller concatenates all non-empty
results with ``,`` to form the full filter graph injected into the decode
command.
"""

from __future__ import annotations

from clearvid.app.schemas.models import EnhancementConfig, VideoMetadata


def build_preprocess_filters(
    config: EnhancementConfig,
    metadata: VideoMetadata,
) -> list[str]:
    """Return an ordered list of FFmpeg filter expressions for preprocessing.

    Filters are applied in the following order:
    1. Deinterlace (if interlaced source)
    2. Denoise (nlmeans)
    3. Deblock (for low-bitrate H.264)
    4. Colorspace normalization (→ BT.709)
    """
    filters: list[str] = []
    filters.extend(_deinterlace_filter(config, metadata))
    filters.extend(_denoise_filter(config, metadata))
    filters.extend(_deblock_filter(config, metadata))
    filters.extend(_colorspace_filter(config))
    return filters


# ---------------------------------------------------------------------------
# Individual filter builders
# ---------------------------------------------------------------------------

def _deinterlace_filter(
    config: EnhancementConfig,
    metadata: VideoMetadata,
) -> list[str]:
    """``bwdif`` deinterlace — auto-detects interlaced content via probe."""
    mode = config.preprocess_deinterlace
    if mode == "off":
        return []
    if mode == "auto" and not metadata.is_interlaced:
        return []
    # bwdif gives better quality than yadif (Bob Weaver Deinterlacing Filter)
    return ["bwdif=mode=send_frame:parity=auto:deint=all"]


def _denoise_filter(
    config: EnhancementConfig,
    metadata: VideoMetadata,
) -> list[str]:
    """``nlmeans`` non-local-means denoise with strength adaptive to bitrate."""
    if not config.preprocess_denoise:
        return []

    # Adaptive strength: lower bitrate → stronger denoise
    # Range: s=3 (light) to s=8 (strong)
    strength = _estimate_denoise_strength(metadata)
    # nlmeans: s=denoise_strength, p=patch_size, r=research_window
    return [f"nlmeans=s={strength:.1f}:p=7:r=15"]


def _deblock_filter(
    config: EnhancementConfig,
    metadata: VideoMetadata,
) -> list[str]:
    """``deblock`` for low-bitrate H.264/MPEG content."""
    if not config.preprocess_deblock:
        return []

    # Only effective for block-based codecs at low bitrates
    codec = (metadata.video_codec or "").lower()
    block_codecs = {"h264", "mpeg2video", "mpeg4", "mpeg1video", "msmpeg4v3"}
    if codec not in block_codecs:
        return []

    # Adaptive filter strength based on bitrate
    bpp = _bits_per_pixel(metadata)
    if bpp is None or bpp > 0.15:
        # High bitrate → little block artifact, skip
        return []

    # Stronger filtering for lower bitrate
    if bpp < 0.05:
        alpha, beta = "3", "3"
    elif bpp < 0.10:
        alpha, beta = "2", "2"
    else:
        alpha, beta = "1", "1"

    return [f"deblock=filter=weak:alpha={alpha}:beta={beta}"]


def _colorspace_filter(config: EnhancementConfig) -> list[str]:
    """Normalize color matrix to BT.709 for consistent model input."""
    if not config.preprocess_colorspace_normalize:
        return []
    # Convert whatever input colorspace to BT.709
    # Using colorspace filter: set input as auto-detect, output as BT.709
    return ["colorspace=all=bt709:iall=bt601-6-625:fast=1"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_denoise_strength(metadata: VideoMetadata) -> float:
    """Heuristic denoise strength from video bitrate.

    Lower bits-per-pixel → noisier source → stronger denoise.
    Returns nlmeans ``s`` parameter value (3.0–8.0).
    """
    bpp = _bits_per_pixel(metadata)
    if bpp is None:
        return 4.0  # safe moderate default

    if bpp < 0.03:
        return 8.0
    if bpp < 0.05:
        return 6.0
    if bpp < 0.10:
        return 5.0
    if bpp < 0.20:
        return 4.0
    return 3.0


def _bits_per_pixel(metadata: VideoMetadata) -> float | None:
    """Compute bits-per-pixel from bitrate and resolution/fps."""
    if not metadata.bit_rate or metadata.width <= 0 or metadata.height <= 0 or metadata.fps <= 0:
        return None
    pixels_per_second = metadata.width * metadata.height * metadata.fps
    return metadata.bit_rate / pixels_per_second
