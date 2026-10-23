from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min() if values.size else 0.0
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def rank_candidates(
    image_paths: list[str] | np.ndarray,
    probabilities: np.ndarray,
    diversity: np.ndarray | None = None,
    boundary: np.ndarray | None = None,
    balance: np.ndarray | None = None,
    outlier: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank unlabeled samples without selecting them automatically for training."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(image_paths) != len(probabilities):
        raise ValueError("image_paths and probabilities must have matching rows")
    defaults = {"uncertainty": 0.45, "diversity": 0.25, "boundary": 0.15, "balance": 0.15, "outlier": 0.20}
    weights = {**defaults, **(weights or {})}
    ordered = np.sort(probabilities, axis=1)
    top1 = ordered[:, -1]
    top2 = ordered[:, -2] if probabilities.shape[1] > 1 else np.zeros(len(top1))
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    uncertainty = 0.5 * minmax(entropy) + 0.5 * minmax(1.0 - (top1 - top2))
    size = len(image_paths)
    component = lambda values: np.zeros(size) if values is None else minmax(np.asarray(values))
    score = (
        weights["uncertainty"] * uncertainty
        + weights["diversity"] * component(diversity)
        + weights["boundary"] * component(boundary)
        + weights["balance"] * component(balance)
        - weights["outlier"] * component(outlier)
    )
    result = pd.DataFrame({
        "image_path": [str(path) for path in image_paths],
        "predicted_class_index": probabilities.argmax(axis=1),
        "confidence": top1,
        "entropy": entropy,
        "margin": top1 - top2,
        "active_learning_score": score,
    })
    return result.sort_values("active_learning_score", ascending=False).reset_index(drop=True).assign(
        rank=lambda frame: np.arange(1, len(frame) + 1)
    )


def save_ranking(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)