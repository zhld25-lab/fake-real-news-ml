from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix


def save_roc_curve(model, x_test, y_test, output_path: str | Path) -> None:
    """Save ROC curve for a fitted classifier pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve - Final Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_plot(model, x_test, y_test, output_path: str | Path) -> None:
    """Save a confusion matrix plot for a fitted classifier pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Fake", "Real"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix - Final Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
