import os
import subprocess
import sys

from config import BASE_DIR, SRC_DIR


def run_step(name: str, script: str):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}")
    result = subprocess.run([sys.executable, os.path.join(SRC_DIR, script)],
                            cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"ERROR: {name} failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    steps = [
        ("1. Preprocessing", "preprocess.py"),
        ("2. Generate Teacher Labels", "generate_labels.py"),
        ("3. Train LightUNet", "train.py"),
        ("4. Evaluate & Compare", "test.py"),
    ]
    for name, script in steps:
        run_step(name, script)

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE!")
    print(f"  Check results/ and checkpoints/ for outputs.")
    print(f"{'=' * 60}")
