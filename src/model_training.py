from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluation import evaluate_predictions, model_scores


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 3
    max_features: int = 30000
    ngram_min: int = 1
    ngram_max: int = 2


def split_dataset(df: pd.DataFrame, config: TrainingConfig):
    """Create a stratified train/test split from a cleaned dataframe."""
    return train_test_split(
        df["clean_text"],
        df["label"],
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label"],
    )


def build_pipeline(classifier, config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=config.max_features,
                    ngram_range=(config.ngram_min, config.ngram_max),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def candidate_models(include_stacking: bool = False) -> dict[str, object]:
    models: dict[str, object] = {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }

    if include_stacking:
        models["Stacking"] = StackingClassifier(
            estimators=[
                ("nb", MultinomialNB(alpha=0.5)),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
                ("svm", LinearSVC(class_weight="balanced", random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=1200, random_state=42),
            cv=3,
            n_jobs=-1,
        )

    return models


def run_model_comparison(
    x_train,
    x_test,
    y_train,
    y_test,
    config: TrainingConfig,
    include_stacking: bool = False,
):
    """Train candidate traditional ML models and return a comparison table."""
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    scorer = make_scorer(f1_score, average="macro", zero_division=0)
    rows = []
    fitted_models = {}

    for model_name, classifier in candidate_models(include_stacking).items():
        pipeline = build_pipeline(clone(classifier), config)
        cv_scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        y_score = model_scores(pipeline, x_test)
        metrics = evaluate_predictions(y_test, y_pred, y_score)

        rows.append(
            {
                "Model": model_name,
                "CV_Macro_F1_Mean": float(cv_scores.mean()),
                "CV_Macro_F1_Std": float(cv_scores.std()),
                "Test_Accuracy": metrics["accuracy"],
                "Test_Macro_F1": metrics["macro_f1"],
                "Test_Weighted_F1": metrics["weighted_f1"],
                "Test_AUC": metrics["auc"],
            }
        )
        fitted_models[model_name] = pipeline

    results = pd.DataFrame(rows).sort_values(
        by="CV_Macro_F1_Mean",
        ascending=False,
    )
    return results.reset_index(drop=True), fitted_models


def choose_best_model(results: pd.DataFrame, fitted_models: dict[str, Pipeline]):
    eligible = results[results["Model"] != "Dummy baseline"].copy()
    if eligible.empty:
        eligible = results.copy()
    best_name = eligible.sort_values(
        by=["CV_Macro_F1_Mean", "Test_Macro_F1"],
        ascending=False,
    ).iloc[0]["Model"]
    return best_name, fitted_models[best_name]
