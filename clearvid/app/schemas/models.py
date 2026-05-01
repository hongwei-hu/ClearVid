from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class QualityMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class BackendType(str, Enum):
    AUTO = "auto"
    BASELINE = "baseline"
    REALESRGAN = "realesrgan"


class TargetProfile(str, Enum):
    SOURCE = "source"
    FHD = "fhd"
    UHD4K = "uhd4k"
    SCALE2X = "scale2x"
    SCALE4X = "scale4x"


class UpscaleModel(str, Enum):
    AUTO = "auto"
    GENERAL_V3 = "general_v3"
    X4PLUS = "x4plus"


class InferenceAccelerator(str, Enum):
    NONE = "none"
    AUTO = "auto"
    COMPILE = "compile"
    TENSORRT = "tensorrt"


class FaceRestoreModel(str, Enum):
    CODEFORMER = "codeformer"
    GFPGAN = "gfpgan"


class HardwareProfile(str, Enum):
    AUTO = "auto"
    HIGH_END = "high_end"
    MID_RANGE = "mid_range"
    LOW_VRAM = "low_vram"


class StreamInfo(BaseModel):
    index: int
    codec_type: str
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    sample_rate: str | None = None
    language: str | None = None


class VideoMetadata(BaseModel):
    input_path: Path
    container: str | None = None
    duration_seconds: float
    bit_rate: int | None = None
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None = None
    audio_streams: int = 0
    subtitle_streams: int = 0
    is_interlaced: bool = False
    streams: list[StreamInfo] = Field(default_factory=list)
    color_primaries: str | None = None  # e.g. "bt709", "smpte170m"; None = unspecified
    color_space: str | None = None      # color matrix, e.g. "bt709", "smpte170m"

    @computed_field
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class EnvironmentInfo(BaseModel):
    ffmpeg_available: bool
    ffprobe_available: bool
    nvidia_smi_available: bool
    ffmpeg_version: str | None = None
    ffmpeg_hwaccels: list[str] = Field(default_factory=list)
    ffmpeg_encoders: list[str] = Field(default_factory=list)
    gpu_name: str | None = None
    gpu_driver_version: str | None = None
    gpu_memory_mb: int | None = None
    torch_version: str | None = None
    torch_cuda_available: bool = False
    torch_gpu_compatible: bool = False
    preferred_backend: BackendType = BackendType.BASELINE
    realesrgan_available: bool = False
    realesrgan_message: str | None = None


class EnhancementConfig(BaseModel):
    input_path: Path
    output_path: Path
    target_profile: TargetProfile = TargetProfile.FHD
    quality_mode: QualityMode = QualityMode.QUALITY
    backend: BackendType = BackendType.AUTO
    upscale_model: UpscaleModel = UpscaleModel.AUTO
    face_restore_enabled: bool = False
    face_restore_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    face_restore_model: FaceRestoreModel = FaceRestoreModel.CODEFORMER
    face_poisson_blend: bool = False
    temporal_stabilize_enabled: bool = False
    temporal_stabilize_strength: float = Field(default=0.6, ge=0.0, le=1.0)
    preprocess_denoise: bool = False
    preprocess_deblock: bool = True
    preprocess_deinterlace: str = Field(default="auto", pattern=r"^(auto|off)$")
    preprocess_colorspace_normalize: bool = True
    denoise_strength: float = Field(default=0.08, ge=0.0, le=1.0)
    sharpen_enabled: bool = True
    sharpen_strength: float = Field(default=0.12, ge=0.0, le=1.0)
    color_correction_enabled: bool = False
    tile_size: int = 0
    tile_pad: int = 16
    batch_size: int = 4
    fp16_enabled: bool = True
    inference_accelerator: InferenceAccelerator = InferenceAccelerator.AUTO
    trt_build_timeout: int | None = None
    skip_frame_threshold: float = Field(default=0.0, ge=0.0, le=20.0)
    async_pipeline: bool = True
    preserve_audio: bool = True
    preserve_subtitles: bool = True
    preserve_metadata: bool = True
    encoder: str = "hevc_nvenc"
    encoder_preset: str = "p5"
    encoder_crf: int | None = None
    video_bitrate: str | None = None
    output_pixel_format: str = "yuv420p"
    hardware_profile: HardwareProfile = HardwareProfile.AUTO
    preview_seconds: int | None = None
    dry_run: bool = False


class ExecutionPlan(BaseModel):
    command: list[str] = Field(default_factory=list)
    output_width: int
    output_height: int
    backend: BackendType
    notes: list[str] = Field(default_factory=list)


class BatchResult(BaseModel):
    input_path: Path
    output_path: Path
    success: bool
    message: str
    backend: BackendType | None = None
