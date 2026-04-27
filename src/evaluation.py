from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def evaluate_predictions(y_true, y_pred, y_score=None) -> dict[str, float | None]:
    """Compute core classification metrics."""
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "fake_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "real_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc": None,
    }
    if y_score is not None:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def classification_report_text(y_true, y_pred) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=["Fake News", "Real News"],
        digits=4,
        zero_division=0,
    )


def confusion_matrix_frame(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["Actual Fake", "Actual Real"],
        columns=["Predicted Fake", "Predicted Real"],
    )


def model_scores(model, x_values):
    """Return score values for the positive class when the model supports it."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_values)
        classes = list(getattr(model, "classes_", []))
        class_index = classes.index(1) if 1 in classes else 1
        return probabilities[:, class_index]

    if hasattr(model, "decision_function"):
        return model.decision_function(x_values)

    return None


def bootstrap_confidence_interval(
    y_true,
    y_pred,
    metric_func: Callable,
    n_rounds: int = 300,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap mean and 95% confidence interval for a metric."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    rng = np.random.default_rng(random_state)
    scores = []

    for _ in range(n_rounds):
        sample_index = rng.integers(0, len(y_true_array), len(y_true_array))
        scores.append(metric_func(y_true_array[sample_index], y_pred_array[sample_index]))

    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
