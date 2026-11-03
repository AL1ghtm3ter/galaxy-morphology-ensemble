"""Metrics for continuous vote-fraction targets; no probability clipping."""
import numpy as np

from .tasks import NUM_TARGETS, task_slices


def regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != NUM_TARGETS:
        raise ValueError(f"Expected targets with shape (N, {NUM_TARGETS}).")
    if targets.shape != predictions.shape or len(targets) == 0:
        raise ValueError("Predictions must match a nonempty target array.")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Metrics require finite targets and predictions.")
    residual = predictions - targets
    true_centered = targets - targets.mean(axis=1, keepdims=True)
    pred_centered = predictions - predictions.mean(axis=1, keepdims=True)
    pearson_denominator = np.linalg.norm(true_centered, axis=1) * np.linalg.norm(pred_centered, axis=1)
    valid = pearson_denominator > 0
    pearson = np.sum(true_centered[valid] * pred_centered[valid], axis=1) / pearson_denominator[valid]
    cosine = np.sum(targets * predictions, axis=1) / (
        np.linalg.norm(targets, axis=1) * np.linalg.norm(predictions, axis=1) + 1e-8
    )
    r2 = None
    if len(targets) > 1:
        numerator = np.sum(residual ** 2, axis=0)
        denominator = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
        ratios = np.zeros(NUM_TARGETS, dtype=np.float64)
        np.divide(numerator, denominator, out=ratios, where=denominator > 0)
        scores = np.where(denominator > 0, 1 - ratios, np.where(numerator == 0, 1.0, 0.0))
        r2 = float(scores.mean())
    return {
        "samples": len(targets),
        "global_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "global_mae": float(np.mean(np.abs(residual))),
        "root_node_accuracy_percent": float(100 * np.mean(
            targets[:, :3].argmax(axis=1) == predictions[:, :3].argmax(axis=1)
        )),
        "r2_uniform_average": r2,
        "mean_per_image_pearson_correlation": float(pearson.mean()) if valid.any() else None,
        "mean_per_image_cosine_similarity": float(cosine.mean()),
        "task_rmse": {
            name: float(np.sqrt(np.mean(residual[:, columns] ** 2)))
            for name, columns in task_slices()
        },
    }

