from __future__ import annotations

from pathlib import Path


def find_realesrgan_weights(weights_path: Path | None) -> list[Path]:
    if not weights_path or not weights_path.exists():
        return []
    if weights_path.is_file() and weights_path.suffix.lower() == ".pth":
        return [weights_path]
    return sorted(path for path in weights_path.glob("*.pth") if path.is_file())


def inspect_realesrgan_runtime(weights_path: Path | None = None) -> tuple[bool, str, str | None, bool, bool]:
    torch_version: str | None = None
    cuda_available = False
    gpu_compatible = False

    try:
        import torch
    except ImportError as exc:
        return False, f"PyTorch is not installed: {exc}", torch_version, cuda_available, gpu_compatible

    torch_version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())

    try:
        import realesrgan  # noqa: F401
    except ImportError as exc:
        return False, f"Real-ESRGAN is not installed: {exc}", torch_version, cuda_available, gpu_compatible
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            f"Real-ESRGAN 运行时初始化失败: {exc}",
            torch_version,
            cuda_available,
            gpu_compatible,
        )

    if not cuda_available:
        return False, "CUDA is not available in the current PyTorch runtime.", torch_version, False, False

    try:
        capability = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        arch_list = set(torch.cuda.get_arch_list())
        gpu_compatible = required_arch in arch_list
    except Exception:  # noqa: BLE001
        gpu_compatible = False

    if not gpu_compatible:
        return (
            False,
            "Current PyTorch build does not support this GPU architecture. Install a build that supports RTX 5090.",
            torch_version,
            True,
            False,
        )

    discovered_weights = find_realesrgan_weights(weights_path)
    if weights_path and not discovered_weights:
        return (
            False,
            f"Real-ESRGAN 权重未找到，请将 .pth 文件放到: {weights_path}",
            torch_version,
            True,
            True,
        )

    return True, "Real-ESRGAN environment looks available", torch_version, True, True


def validate_realesrgan_environment(weights_path: Path | None = None) -> tuple[bool, str]:
    available, message, _, _, _ = inspect_realesrgan_runtime(weights_path)
    return available, message
