from clearvid.app.pipeline import resolve_output_size
from clearvid.app.schemas.models import TargetProfile


def test_resolve_output_size_fhd() -> None:
    width, height = resolve_output_size(854, 480, TargetProfile.FHD)
    assert (width, height) == (1920, 1080)


def test_resolve_output_size_scale4x() -> None:
    width, height = resolve_output_size(854, 480, TargetProfile.SCALE4X)
    assert (width, height) == (3416, 1920)
