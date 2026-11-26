#!/usr/bin/env python3
"""Check the project folders and whether TensorFlow can train on GPU or CPU.

Run from the project directory:
    python check_environment.py
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path


# Output directories required by the project.
REQUIRED_OUTPUT_DIRS = (
    "confusion_matrices",
    "data",
    "histories",
    "models",
    "plots",
    "plots/confusion_matrices",
    "plots/history",
    "reports",
    "results",
    "src",
)

# Files/folders that can help identify the project root.
ROOT_MARKERS = (
    "data",
    "src",
    "check_environment.py",
    "Dockerfile",
    "requirements.txt",
)


def find_project_root() -> Path:
    """Locate the project root from the current directory or script location."""
    candidates = [Path.cwd(), Path("/workspace")]

    script_path = globals().get("__file__")
    if script_path:
        candidates.append(Path(script_path).resolve().parent)

    seen = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in seen:
            continue
        seen.add(candidate)

        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate

    raise FileNotFoundError(
        "Could not identify the project root. Run this script from the project "
        "directory, or place it in the project directory."
    )


def check_and_create_output_dirs(project_root: Path) -> bool:
    """Check each required directory and create it if it does not exist."""
    print("\n" + "=" * 72)
    print("PROJECT OUTPUT DIRECTORIES")
    print("=" * 72)

    all_ok = True

    for relative_path in REQUIRED_OUTPUT_DIRS:
        directory = project_root / relative_path

        if directory.is_dir():
            print(f"[OK]      Exists:  {relative_path}/")
            continue

        if directory.exists():
            print(f"[ERROR]   Path exists but is not a directory: {relative_path}")
            all_ok = False
            continue

        try:
            directory.mkdir(parents=True, exist_ok=True)

            if directory.is_dir():
                print(f"[CREATED] Created: {relative_path}/")
            else:
                print(f"[ERROR]   Could not create: {relative_path}/")
                all_ok = False

        except OSError as exc:
            print(f"[ERROR]   Could not create {relative_path}/: {exc}")
            all_ok = False

    return all_ok


def check_optional_packages() -> None:
    """Report useful package versions without blocking the GPU/CPU test."""
    print("\n" + "=" * 72)
    print("PYTHON PACKAGES")
    print("=" * 72)

    packages = (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("TensorFlow", "tensorflow"),
    )

    for display_name, module_name in packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "version not reported")
            print(f"[OK] {display_name}: {version}")
        except Exception as exc:
            print(f"[WARNING] Could not import {display_name}: {exc}")


def test_tensorflow_training(device_name: str) -> tuple[bool, str]:
    """Run a small matrix calculation and CNN training step on the chosen device."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # First verify that a calculation is actually placed on the requested
        # device. This avoids treating silent CPU placement as a GPU success.
        with tf.device(device_name):
            a = tf.random.uniform((256, 256))
            b = tf.random.uniform((256, 256))
            result = tf.matmul(a, b)
            _ = float(tf.reduce_mean(result).numpy())
            result_device = result.device.upper()

        expected_device = "GPU" if "GPU" in device_name.upper() else "CPU"

        if expected_device not in result_device:
            return (
                False,
                f"TensorFlow placed the matrix calculation on {result.device}, "
                f"not the requested {device_name}.",
            )

        # Then test a small, real model training step.
        with tf.device(device_name):
            model = keras.Sequential(
                [
                    layers.Input(shape=(48, 48, 3)),
                    layers.Conv2D(8, 3, activation="relu"),
                    layers.MaxPooling2D(),
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(7, activation="softmax"),
                ]
            )

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            images = tf.random.uniform((4, 48, 48, 3))
            labels = tf.one_hot([0, 1, 2, 3], depth=7)

            start_time = time.perf_counter()
            model.train_on_batch(images, labels)
            elapsed = time.perf_counter() - start_time

        del model
        keras.backend.clear_session()

        return (
            True,
            f"Matrix calculation and CNN training step succeeded on "
            f"{device_name} ({elapsed:.2f} seconds).",
        )

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_gpu_then_cpu() -> bool:
    """Try GPU first; if it fails, try CPU and report the outcome."""
    print("\n" + "=" * 72)
    print("TENSORFLOW DEVICE TEST")
    print("=" * 72)

    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"[ERROR] TensorFlow could not be imported: {exc}")
        print("[ERROR] GPU and CPU training tests cannot run.")
        return False

    print(f"TensorFlow version: {tf.__version__}")

    try:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"TensorFlow GPU devices detected: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
    except Exception as exc:
        print(f"[WARNING] Could not list GPU devices: {exc}")

    # GPU is always attempted first, including when TensorFlow reports no GPU.
    print("\nAttempting GPU calculation and CNN training...")
    gpu_ok, gpu_message = test_tensorflow_training("/GPU:0")

    if gpu_ok:
        print(f"[OK] {gpu_message}")
        return True

    print(f"[GPU FAILED] {gpu_message}")
    print("Attempting CPU calculation and CNN training...")

    cpu_ok, cpu_message = test_tensorflow_training("/CPU:0")

    if cpu_ok:
        print(f"[OK] {cpu_message}")
        print(
            "[WARNING] TensorFlow is not running successfully on the GPU, "
            "but it can run on the CPU."
        )
        return True

    print(f"[CPU FAILED] {cpu_message}")
    print("[ERROR] TensorFlow could not complete the test on either GPU or CPU.")
    return False


def main() -> int:
    print("Environment check starting...")
    print(f"Current working directory: {Path.cwd()}")

    try:
        project_root = find_project_root()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    print(f"Project root: {project_root}")

    directories_ok = check_and_create_output_dirs(project_root)
    check_optional_packages()
    compute_ok = check_gpu_then_cpu()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"[{'PASS' if directories_ok else 'FAIL'}] Required output directories")
    print(f"[{'PASS' if compute_ok else 'FAIL'}] TensorFlow GPU/CPU execution")

    if not directories_ok or not compute_ok:
        print("\n[ERROR] Environment check failed.")
        return 1

    print("\n[PASS] Environment check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
