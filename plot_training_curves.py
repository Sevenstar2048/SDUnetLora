from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
SD_LOG_DIR = ROOT / "finetune/lora/digimon/logs"
FLUX_LOG_DIR = ROOT / "finetune/lora/flux_digimon/logs"
OUT_DIR = ROOT / "training_curves"

# From trainLoRA.py (used only when SD lr was not logged).
SD_BASE_LR = 1e-4
SD_MAX_TRAIN_STEPS = 1000
SD_WARMUP_STEPS = 0


def list_event_files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.rglob("events.out.tfevents.*"))


def read_scalar_from_events(log_dir: Path, scalar_tag: str) -> tuple[list[int], list[float]]:
    """Merge same scalar tag from multiple event files, sorted by global step."""
    event_files = list_event_files(log_dir)
    merged: dict[int, float] = {}

    for file in event_files:
        try:
            acc = EventAccumulator(str(file), size_guidance={"scalars": 0})
            acc.Reload()
            if scalar_tag not in acc.Tags().get("scalars", []):
                continue
            for ev in acc.Scalars(scalar_tag):
                merged[int(ev.step)] = float(ev.value)
        except Exception:
            # Skip corrupted/incompatible event files.
            continue

    steps = sorted(merged.keys())
    values = [merged[s] for s in steps]
    return steps, values


def cosine_lr_curve(base_lr: float, max_steps: int, warmup_steps: int) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []

    for step in range(1, max_steps + 1):
        if warmup_steps > 0 and step <= warmup_steps:
            lr = base_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
        steps.append(step)
        values.append(lr)

    return steps, values


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sd_loss_steps, sd_loss_vals = read_scalar_from_events(SD_LOG_DIR, "train_loss")
    sd_lr_steps, sd_lr_vals = read_scalar_from_events(SD_LOG_DIR, "lr")

    flux_loss_steps, flux_loss_vals = read_scalar_from_events(FLUX_LOG_DIR, "loss")
    flux_lr_steps, flux_lr_vals = read_scalar_from_events(FLUX_LOG_DIR, "lr")

    # SD run does not log lr by default in this project; fall back to theoretical scheduler curve.
    sd_lr_is_theoretical = False
    if not sd_lr_steps:
        sd_lr_steps, sd_lr_vals = cosine_lr_curve(
            base_lr=SD_BASE_LR,
            max_steps=SD_MAX_TRAIN_STEPS,
            warmup_steps=SD_WARMUP_STEPS,
        )
        sd_lr_is_theoretical = True

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("SD / Flux Training Curves", fontsize=16)

    ax = axes[0, 0]
    if sd_loss_steps:
        ax.plot(sd_loss_steps, sd_loss_vals, color="#1f77b4", linewidth=1.8)
        ax.set_title("SD LoRA Loss (train_loss)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No SD loss found", ha="center", va="center")
        ax.set_title("SD LoRA Loss")

    ax = axes[0, 1]
    if sd_lr_steps:
        ax.plot(sd_lr_steps, sd_lr_vals, color="#ff7f0e", linewidth=1.8)
        suffix = " (theoretical cosine)" if sd_lr_is_theoretical else " (logged)"
        ax.set_title(f"SD LoRA LR{suffix}")
        ax.set_xlabel("step")
        ax.set_ylabel("lr")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No SD lr found", ha="center", va="center")
        ax.set_title("SD LoRA LR")

    ax = axes[1, 0]
    if flux_loss_steps:
        ax.plot(flux_loss_steps, flux_loss_vals, color="#2ca02c", linewidth=1.8)
        ax.set_title("Flux LoRA Loss (loss)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No Flux loss found", ha="center", va="center")
        ax.set_title("Flux LoRA Loss")

    ax = axes[1, 1]
    if flux_lr_steps:
        ax.plot(flux_lr_steps, flux_lr_vals, color="#d62728", linewidth=1.8)
        ax.set_title("Flux LoRA LR (logged)")
        ax.set_xlabel("step")
        ax.set_ylabel("lr")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No Flux lr found", ha="center", va="center")
        ax.set_title("Flux LoRA LR")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_png = OUT_DIR / "sd_flux_training_curves.png"
    out_pdf = OUT_DIR / "sd_flux_training_curves.pdf"
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("=== Curve Generation Done ===")
    print(f"Output PNG: {out_png.as_posix()}")
    print(f"Output PDF: {out_pdf.as_posix()}")
    print(f"SD loss points: {len(sd_loss_steps)}")
    print(f"SD lr points: {len(sd_lr_steps)} {'(theoretical)' if sd_lr_is_theoretical else '(logged)'}")
    print(f"Flux loss points: {len(flux_loss_steps)}")
    print(f"Flux lr points: {len(flux_lr_steps)}")


if __name__ == "__main__":
    main()
