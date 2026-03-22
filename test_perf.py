"""Quick performance benchmark: batch vs single-frame Real-ESRGAN."""
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    import torch

    from clearvid.app.models.realesrgan_runner import (
        _enhance_frames_batch,
        _load_runtime_components,
        _resolve_tile_size,
    )

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory // (1024 * 1024)} MB")

    tile = _resolve_tile_size(0, 854, 480, True)
    print(f"Resolved tile for 480p: {tile}")

    cls, srvgg, _ = _load_runtime_components()
    model = srvgg(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
    upsampler = cls(
        scale=4,
        model_path="weights/realesrgan/realesr-general-x4v3.pth",
        model=model,
        tile=tile,
        tile_pad=16,
        pre_pad=0,
        half=True,
    )
    print(f"tile_size={upsampler.tile_size}")

    dummy = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)

    print("Warmup...")
    upsampler.enhance(dummy, outscale=2.0)
    torch.cuda.synchronize()

    print("--- Single x4 ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(4):
        upsampler.enhance(dummy, outscale=2.0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s total, {(t1 - t0) / 4:.3f}s/frame")

    print("--- Batch(4) ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = _enhance_frames_batch([dummy] * 4, upsampler, 2.0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s total, {(t1 - t0) / 4:.3f}s/frame, shape={results[0].shape}")

    print("--- Batch(8) ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = _enhance_frames_batch([dummy] * 8, upsampler, 2.0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s total, {(t1 - t0) / 8:.3f}s/frame")

    print("ALL OK")


if __name__ == "__main__":
    main()
