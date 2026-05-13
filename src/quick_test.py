"""Quick smoke test: small dataset, few epochs. Verifies pipeline integrity in ~8 min."""
import os
import subprocess
import sys

from config import BASE_DIR, SRC_DIR

SMOKE_IMAGES = 32   # 8 per category
SMOKE_EPOCHS = 5


def run_step(name: str, script: str, extra_args: list[str] | None = None):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}")
    cmd = [sys.executable, os.path.join(SRC_DIR, script)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"ERROR: {name} failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("  QUICK SMOKE TEST")
    print(f"  {SMOKE_IMAGES} images, {SMOKE_EPOCHS} epochs, ~8 minutes")
    print("=" * 60)

    steps = [
        ("1. Preprocessing", "preprocess.py", ["--max-total", str(SMOKE_IMAGES)]),
        ("2. Generate Teacher Labels", "generate_labels.py", None),
        ("3. Train LightUNet", "train.py", ["--epochs", str(SMOKE_EPOCHS)]),
        ("4. Evaluate & Compare", "test.py", None),
    ]
    for name, script, extra_args in steps:
        run_step(name, script, extra_args)

    print(f"\n{'=' * 60}")
    print("  SMOKE TEST PASSED!")
    print(f"  Pipeline is healthy. Run 'python src/run_pipeline.py' for full training.")
    print(f"{'=' * 60}")
