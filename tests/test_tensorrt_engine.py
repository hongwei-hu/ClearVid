from __future__ import annotations

from clearvid.app.models.tensorrt_engine import _read_engine_profile_shapes


class _TensorProfileEngine:
    def get_tensor_profile_shape(self, tensor_name: str, profile_index: int):
        assert tensor_name == "input"
        assert profile_index == 0
        return ((1, 3, 128, 128), (4, 3, 512, 512), (4, 3, 544, 544))


class _LegacyProfileEngine:
    def get_profile_shape(self, tensor_name: str, profile_index: int):
        assert tensor_name == "input"
        assert profile_index == 0
        return ((1, 3, 128, 128), (2, 3, 512, 512), (2, 3, 544, 544))


def test_read_engine_profile_shapes_prefers_tensor_api() -> None:
    shapes = _read_engine_profile_shapes(_TensorProfileEngine(), "input", 0)

    assert shapes[1] == (4, 3, 512, 512)
    assert shapes[2][0] == 4


def test_read_engine_profile_shapes_supports_legacy_api() -> None:
    shapes = _read_engine_profile_shapes(_LegacyProfileEngine(), "input", 0)

    assert shapes[1] == (2, 3, 512, 512)
    assert shapes[2][0] == 2
