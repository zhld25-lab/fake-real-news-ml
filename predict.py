from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.data_utils import LABEL_NAMES
from src.evaluation import model_scores
from src.text_preprocessing import clean_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a news text is fake or real.")
    parser.add_argument("--model-path", default="models/final_model.pkl")
    parser.add_argument("--text", default=None, help="Headline or article text to classify.")
    parser.add_argument("--file", default=None, help="Optional text file containing an article.")
    return parser.parse_args()


def read_input(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    raise ValueError("Please provide --text or --file.")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} was not found. Train a model first with: python train_model.py"
        )

    model = joblib.load(model_path)
    text = read_input(args)
    cleaned = clean_text(text)
    prediction = int(model.predict([cleaned])[0])
    score = model_scores(model, [cleaned])

    print(f"Predicted label: {LABEL_NAMES.get(prediction, prediction)}")
    print(f"Numeric label: {prediction}")
    if score is not None:
        value = float(score[0])
        if hasattr(model, "predict_proba"):
            confidence = value if prediction == 1 else 1 - value
            print(f"Confidence: {confidence:.4f}")
            print(f"Real-news probability: {value:.4f}")
        else:
            print(f"Decision score: {value:.4f}")
    print("\nNote: this is a text-pattern classifier, not a factual verification system.")


if __name__ == "__main__":
    main()
