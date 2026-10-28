"""Resolve canonical model checkpoints across a checkout and local training runs."""

from pathlib import Path


MODELS = ("baseline", "meanpool", "compact_gru", "gru")
SEEDS = (42, 43, 44)


def resolve_checkpoint_dir(root: str | Path = ".") -> Path:
    """Return the complete preserved M4 snapshot, falling back to local run outputs.

    The repository's canonical checkpoint files live in ``experiments/m4_final``.
    ``results/checkpoints`` is a local training-output location and may be ignored
    by Git, so it is only a fallback for incomplete development checkouts.
    """
    root = Path(root).resolve()
    candidates = (
        root / "experiments" / "m4_final" / "checkpoints",
        root / "results" / "checkpoints",
    )
    for candidate in candidates:
        if all((candidate / model / f"best_seed{seed}.pt").is_file() for model in MODELS for seed in SEEDS):
            return candidate
    expected = [str(candidate / model / f"best_seed{seed}.pt") for candidate in candidates for model in MODELS for seed in SEEDS]
    raise FileNotFoundError(
        "No complete checkpoint set found. Expected all baseline, meanpool, compact_gru, "
        "and gru seed 42/43/44 files under experiments/m4_final/checkpoints or "
        "results/checkpoints. First expected path: " + expected[0]
    )
