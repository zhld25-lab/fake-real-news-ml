from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data_utils import dataset_summary, load_news_data
from src.evaluation import (
    bootstrap_confidence_interval,
    classification_report_text,
    confusion_matrix_frame,
    model_scores,
)
from src.model_training import (
    TrainingConfig,
    choose_best_model,
    run_model_comparison,
    split_dataset,
)
from src.text_preprocessing import add_clean_text_column
from src.visualization import save_confusion_matrix_plot, save_roc_curve


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train traditional ML models for fake vs real news classification."
    )
    parser.add_argument("--fake-path", default="data/Fake.csv")
    parser.add_argument("--true-path", default="data/True.csv")
    parser.add_argument("--model-path", default="models/final_model.pkl")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--include-stacking",
        action="store_true",
        help="Also train a stacking classifier. This is slower but still traditional ML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    model_path = Path(args.model_path)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print_section("1. Load data")
    print("Reading data/Fake.csv and data/True.csv.")
    raw_df = load_news_data(args.fake_path, args.true_path)
    if args.sample_size:
        samples = []
        per_class = max(1, args.sample_size // raw_df["label"].nunique())
        for _, group in raw_df.groupby("label"):
            samples.append(group.sample(min(len(group), per_class), random_state=42))
        raw_df = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Using a stratified sample for a faster run: {len(raw_df):,} rows")
    print(pd.Series(dataset_summary(raw_df)).to_string())

    print_section("2. Clean text")
    print(
        "Combining title + article text, removing URLs/HTML, lowercasing, "
        "keeping only letters, and trimming extra spaces."
    )
    df = add_clean_text_column(raw_df)
    print(f"Rows after cleaning empty text: {len(df):,}")
    print(df[["title", "label_name", "clean_text"]].head(3).to_string(index=False))

    print_section("3. Train/Test split")
    config = TrainingConfig(
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        max_features=args.max_features,
    )
    x_train, x_test, y_train, y_test = split_dataset(df, config)
    print(f"Training rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print("Train label counts:")
    print(y_train.value_counts().sort_index().rename(index={0: "Fake", 1: "Real"}).to_string())
    print("Test label counts:")
    print(y_test.value_counts().sort_index().rename(index={0: "Fake", 1: "Real"}).to_string())

    print_section("4. TF-IDF vectorization")
    print(
        "TF-IDF is inside each sklearn Pipeline. It is fitted only on training folds "
        "during cross-validation, which helps avoid data leakage."
    )
    print(f"TF-IDF max_features={config.max_features}, ngram_range=(1, 2)")

    print_section("5. Cross validation training")
    print(f"Training candidate traditional ML models with {config.cv_folds}-fold CV.")
    results_df, fitted_models = run_model_comparison(
        x_train,
        x_test,
        y_train,
        y_test,
        config,
        include_stacking=args.include_stacking,
    )
    print(results_df.round(4).to_string(index=False))

    print_section("6. Hyperparameter tuning (GridSearchCV)")
    print(
        "This script uses fixed, reviewable hyperparameters for reproducibility. "
        "For a course report, document the selected values and compare them with "
        "the notebook's GridSearchCV experiments."
    )

    print_section("7. Model comparison (F1)")
    best_model_name, best_model = choose_best_model(results_df, fitted_models)
    print("Models are ranked by cross-validation Macro F1, not by test F1.")
    print(results_df[["Model", "CV_Macro_F1_Mean", "Test_Macro_F1"]].round(4).to_string(index=False))
    print(f"Best model selected: {best_model_name}")
    results_df.to_csv(outputs_dir / "model_comparison.csv", index=False)

    print_section("8. Retrain best model on full training data")
    best_model.fit(x_train, y_train)
    joblib.dump(best_model, model_path)
    print(f"Saved trained pipeline to: {model_path}")

    print_section("9. Final evaluation on test set")
    y_pred = best_model.predict(x_test)
    y_score = model_scores(best_model, x_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    if y_score is not None:
        print("AUC is available because the selected model exposes probabilities or scores.")
    else:
        print("AUC is not available for the selected model output.")

    report = classification_report_text(y_test, y_pred)
    matrix_df = confusion_matrix_frame(y_test, y_pred)
    print("\nClassification report:")
    print(report)
    print("\nConfusion matrix:")
    print(matrix_df.to_string())

    save_text(outputs_dir / "classification_report.txt", report)
    matrix_df.to_csv(outputs_dir / "confusion_matrix.csv")

    try:
        save_confusion_matrix_plot(best_model, x_test, y_test, outputs_dir / "confusion_matrix.png")
        if y_score is not None:
            save_roc_curve(best_model, x_test, y_test, outputs_dir / "roc_curve.png")
        print(f"Saved evaluation images to: {outputs_dir}")
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")

    print_section("Statistical component - Bootstrap confidence intervals")
    acc_mean, acc_low, acc_high = bootstrap_confidence_interval(
        y_test,
        y_pred,
        accuracy_score,
        n_rounds=300,
    )
    f1_mean, f1_low, f1_high = bootstrap_confidence_interval(
        y_test,
        y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        n_rounds=300,
    )
    bootstrap_text = (
        f"Best Model: {best_model_name}\n\n"
        "Bootstrap 300 rounds - 95% Confidence Intervals\n\n"
        f"Accuracy: mean={acc_mean:.4f}, 95% CI=({acc_low:.4f}, {acc_high:.4f})\n"
        f"Macro F1: mean={f1_mean:.4f}, 95% CI=({f1_low:.4f}, {f1_high:.4f})\n"
    )
    print(bootstrap_text)
    save_text(outputs_dir / "bootstrap_confidence_intervals.txt", bootstrap_text)

    print_section("Final summary")
    print(f"Final model: {best_model_name}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final Macro F1-score: {macro_f1:.4f}")
    print(f"Saved model: {model_path}")
    print("Script finished successfully.")


if __name__ == "__main__":
    main()
