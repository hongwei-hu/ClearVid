from __future__ import annotations

from pathlib import Path

from clearvid.app.models import stream_codec
from clearvid.app.schemas.models import EnhancementConfig, VideoMetadata


def _make_config(**kwargs) -> EnhancementConfig:
    defaults = {"input_path": Path("in.mp4"), "output_path": Path("out.mp4")}
    defaults.update(kwargs)
    return EnhancementConfig(**defaults)


def _make_metadata(**kwargs) -> VideoMetadata:
    defaults = {
        "input_path": Path("in.mp4"),
        "duration_seconds": 60.0,
        "width": 640,
        "height": 360,
        "fps": 29.97,
        "video_codec": "h264",
    }
    defaults.update(kwargs)
    return VideoMetadata(**defaults)


def test_build_decode_command_includes_preview_and_filters() -> None:
    cfg = _make_config(preview_seconds=12, preprocess_denoise=True, denoise_strength=0.4)
    cmd = stream_codec.build_decode_command(cfg, _make_metadata())

    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "12"
    assert "-vf" in cmd
    assert "nlmeans" in cmd[cmd.index("-vf") + 1]
    assert cmd[-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]


def test_build_encode_command_prefers_bitrate_over_default_cq(tmp_path: Path) -> None:
    cfg = _make_config(video_bitrate="8M", encoder_crf=None)
    out = tmp_path / "enhanced.mp4"
    cmd = stream_codec.build_encode_command(cfg, _make_metadata(), 1920, 1080, out)

    assert "-b:v" in cmd
    assert cmd[cmd.index("-b:v") + 1] == "8M"
    assert "-cq" not in cmd
    assert cmd[-2:] == ["-an", str(out)]


def test_build_encode_command_adds_av1_tier(tmp_path: Path) -> None:
    cfg = _make_config(encoder="av1_nvenc", encoder_preset="p6", encoder_crf=22)
    cmd = stream_codec.build_encode_command(cfg, _make_metadata(fps=60.0), 3840, 2160, tmp_path / "tmp.mp4")

    assert cmd[cmd.index("-preset") + 1] == "p6"
    assert "-tier" in cmd
    assert cmd[cmd.index("-cq") + 1] == "22"
    assert cmd[cmd.index("-r") + 1] == "60.000000"


def test_mux_output_maps_optional_streams_and_metadata(tmp_path: Path) -> None:
    recorded: list[str] = []
    cfg = _make_config(preserve_audio=True, preserve_subtitles=True, preserve_metadata=False)

    stream_codec.mux_output(cfg, tmp_path / "video.mp4", command_runner=recorded.extend)

    assert "-map" in recorded
    assert "1:a?" in recorded
    assert "1:s?" in recorded
    assert "-map_metadata" in recorded
    assert recorded[recorded.index("-map_metadata") + 1] == "-1"
    assert recorded[-1] == "out.mp4"


def test_mux_preview_returns_false_on_ffmpeg_error(tmp_path: Path) -> None:
    def fail_run(*_args, **_kwargs):
        raise RuntimeError("boom")

    ok = stream_codec.mux_preview(
        _make_config(),
        tmp_path / "temp.mp4",
        tmp_path / "preview.mp4",
        10.0,
        run_factory=fail_run,
    )

    assert ok is False
