from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clearvid.app.schemas.models import EnvironmentInfo, VideoMetadata


LOW_RES_PIXEL_LIMIT = 921_600
FHD_PIXEL_LIMIT = 2_073_600
LOW_BITRATE_BPP = 0.08


def choose_target_profile(metadata: VideoMetadata, notes: list[str]) -> str:
    pixels = metadata.width * metadata.height
    if pixels <= LOW_RES_PIXEL_LIMIT:
        notes.append(f"输入 {metadata.width}x{metadata.height} 较低，推荐提升至 1080p")
        return "fhd"
    if pixels <= FHD_PIXEL_LIMIT:
        notes.append(f"输入 {metadata.width}x{metadata.height}，推荐提升至 4K")
        return "uhd4k"
    notes.append(f"输入已是 {metadata.width}x{metadata.height} 高分辨率，建议保持原始分辨率")
    return "source"


def choose_quality_mode(vram_mb: int, notes: list[str]) -> str:
    if vram_mb >= 12_000:
        return "quality"
    if vram_mb >= 6_000:
        return "balanced"
    notes.append("显存有限，推荐快速模式以避免 OOM")
    return "fast"


def has_gpu_acceleration(environment: EnvironmentInfo, vram_mb: int) -> bool:
    return (
        environment.torch_gpu_compatible
        and environment.nvidia_smi_available
        and vram_mb >= 6_000
    )


def choose_upscale_model(quality_mode: str, vram_mb: int, gpu_capable: bool, notes: list[str]) -> str:
    if quality_mode == "quality" and vram_mb >= 8_000 and not gpu_capable:
        notes.append("无硬件加速 + 高质量模式，使用 RRDB x4plus 模型以获得最佳细节")
        return "x4plus"
    if quality_mode == "quality" and gpu_capable:
        notes.append("检测到 GPU 加速可用，使用 general_v3 (TRT 加速后性能更优)")
    return "general_v3"


def choose_tile_size(vram_mb: int, notes: list[str]) -> int:
    if vram_mb >= 16_000:
        notes.append("大显存，禁用分块以获得最佳质量")
        return 0
    if vram_mb >= 8_000:
        return 512
    if vram_mb >= 4_000:
        return 256
    notes.append("低显存，使用 128 分块尺寸")
    return 128


def choose_encoder(environment: EnvironmentInfo, vram_mb: int, notes: list[str]) -> str:
    encoders = environment.ffmpeg_encoders
    if "av1_nvenc" in encoders and vram_mb >= 8_000:
        notes.append("检测到 AV1 硬件编码支持，使用 AV1 以获得更高压缩效率")
        return "av1_nvenc"
    if "hevc_nvenc" in encoders:
        return "hevc_nvenc"
    notes.append("未检测到硬件编码器，回退到 CPU 编码 (libx264)")
    return "libx264"


def bitrate_per_pixel(metadata: VideoMetadata) -> float:
    denominator = max(metadata.width * metadata.height * metadata.fps, 1)
    return (metadata.bit_rate or 0) / denominator


def should_denoise(metadata: VideoMetadata) -> tuple[bool, float]:
    bpp_val = bitrate_per_pixel(metadata)
    return bpp_val > 0 and bpp_val < LOW_BITRATE_BPP, bpp_val


def add_high_bitrate_note(metadata: VideoMetadata, bitrate_kbps: int, notes: list[str]) -> None:
    pixels = metadata.width * metadata.height
    if bitrate_kbps > 15_000 and pixels >= FHD_PIXEL_LIMIT:
        notes.append(
            f"输入码率已较高 ({bitrate_kbps} kbps)，超分提升可能有限，"
            "可考虑保持原始分辨率或仅做人脸修复"
        )
