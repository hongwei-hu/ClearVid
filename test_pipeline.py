"""Full pipeline test with timing."""
import sys
import time

sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    from pathlib import Path

    from clearvid.app.orchestrator import Orchestrator
    from clearvid.app.schemas.models import EnhancementConfig, TargetProfile

    config = EnhancementConfig(
        input_path=Path("samples/480P_2000K_306841291pigmask.mp4"),
        output_path=Path("outputs/test_perf_optimized.mp4"),
        target_profile=TargetProfile.FHD,
        preview_seconds=1,
    )

    progress = []

    def on_progress(p: int, m: str) -> None:
        progress.append((p, m))
        print(f"  [{p:3d}%] {m}")

    start = time.perf_counter()
    result = Orchestrator().run_single(config, progress_callback=on_progress)
    elapsed = time.perf_counter() - start

    print(f"\nResult: success={result.success}, backend={result.backend}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Progress events: {len(progress)}")

    # Verify output
    p = Path("outputs/test_perf_optimized.mp4")
    print(f"Output exists: {p.exists()}, size: {p.stat().st_size if p.exists() else 0}")


if __name__ == "__main__":
    main()
