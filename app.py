from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_utils import LABEL_NAMES, dataset_summary, load_news_data
from src.evaluation import model_scores
from src.text_preprocessing import clean_text


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "final_model.pkl"


st.set_page_config(
    page_title="Fake vs Real News Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .site-note {
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        color: #334155;
        margin: 1rem 0;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return load_news_data(DATA_DIR / "Fake.csv", DATA_DIR / "True.csv")


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.write(subtitle)


def model_status_box(model) -> None:
    if model is None:
        st.warning(
            "No trained model file found yet. Train and save the pipeline first with "
            "`python train_model.py`, then refresh this page."
        )
    else:
        st.success("Loaded `models/final_model.pkl`. The interactive classifier is ready.")


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def show_home() -> None:
    page_header(
        "Fake vs Real News Classification using Machine Learning",
        "An interactive academic project website for binary news text classification.",
    )
    st.markdown(
        """
        <div class="site-note">
        This project classifies news articles into two labels: <b>Fake News = 0</b>
        and <b>Real News = 1</b>. It uses traditional machine learning only:
        text cleaning, TF-IDF vectorization, cross-validation, model comparison,
        error analysis, and confidence interval reporting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Task", "Binary Classification")
    col2.metric("Text Features", "TF-IDF")
    col3.metric("Model Type", "Traditional ML")

    st.subheader("Team Members")
    st.write("Xiaoxi Gao and Zhenzhe Luo")

    st.subheader("Motivation")
    st.write(
        "Fake news detection is an important text classification problem because online "
        "news can spread quickly and influence public understanding. This project does "
        "not perform real fact-checking. Instead, it studies whether traditional machine "
        "learning models can learn text-pattern differences between fake and real news "
        "articles in the provided dataset."
    )

    st.subheader("Repository Workflow")
    st.code(
        "pip install -r requirements.txt\n"
        "python train_model.py\n"
        "streamlit run app.py",
        language="bash",
    )


def show_dataset() -> None:
    page_header("Dataset Explorer", "Inspect the official project dataset used by this version.")
    try:
        df = load_dataset()
    except Exception as exc:
        st.error(f"Dataset could not be loaded: {exc}")
        st.info("Expected files: `data/Fake.csv` and `data/True.csv`.")
        return

    summary = dataset_summary(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Columns", len(summary["columns"]))
    col3.metric("Fake Rows", f"{summary['label_counts'].get('Fake News', 0):,}")
    col4.metric("Real Rows", f"{summary['label_counts'].get('Real News', 0):,}")

    st.subheader("Label Mapping")
    st.dataframe(
        pd.DataFrame(
            [
                {"Label Name": "Fake News", "Numeric Label": 0},
                {"Label Name": "Real News", "Numeric Label": 1},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Class Distribution")
    counts = df["label_name"].value_counts().rename_axis("Class").reset_index(name="Count")
    st.bar_chart(counts, x="Class", y="Count", use_container_width=True)

    st.subheader("Data Preview")
    label_filter = st.multiselect(
        "Filter by label",
        options=sorted(df["label_name"].unique()),
        default=sorted(df["label_name"].unique()),
    )
    subject_values = sorted([value for value in df["subject"].dropna().unique() if str(value).strip()])
    subject_filter = st.multiselect("Filter by subject", options=subject_values, default=[])
    preview_df = df[df["label_name"].isin(label_filter)]
    if subject_filter:
        preview_df = preview_df[preview_df["subject"].isin(subject_filter)]
    st.dataframe(
        preview_df[["title", "text", "subject", "date", "label_name"]].head(100),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Dataset design note"):
        st.write(
            "This cleaned project version keeps the official source files as `data/Fake.csv` "
            "and `data/True.csv`. The training code standardizes fields into title, text, "
            "subject, date, label, label_name, and source_file."
        )


def show_methodology() -> None:
    page_header("Methodology", "A transparent traditional machine learning pipeline.")
    steps = pd.DataFrame(
        [
            {
                "Step": "Text cleaning",
                "Purpose": "Remove URLs, HTML tags, non-letter characters, casing differences, and extra spaces.",
            },
            {
                "Step": "Tokenization",
                "Purpose": "Handled by TF-IDF vectorizer after text normalization.",
            },
            {
                "Step": "TF-IDF vectorization",
                "Purpose": "Convert article text into weighted word and phrase features.",
            },
            {
                "Step": "Traditional ML models",
                "Purpose": "Compare Naive Bayes, Logistic Regression, Linear SVM, and optional Stacking.",
            },
            {
                "Step": "Cross-validation",
                "Purpose": "Estimate model performance on training folds before final test evaluation.",
            },
            {
                "Step": "Bootstrap intervals",
                "Purpose": "Estimate uncertainty around final accuracy and Macro F1.",
            },
            {
                "Step": "Error analysis",
                "Purpose": "Review wrong predictions and discuss dataset bias/generalization limits.",
            },
        ]
    )
    st.dataframe(steps, use_container_width=True, hide_index=True)

    st.info(
        "TF-IDF is kept inside the sklearn Pipeline. This is important because each "
        "cross-validation fold fits TF-IDF only on its training portion, reducing data leakage."
    )

    with st.expander("Why no deep learning?"):
        st.write(
            "The project requirement is traditional machine learning only. This version does "
            "not use BERT, Transformers, LSTM, CNN, TensorFlow, or PyTorch."
        )


def show_training() -> None:
    page_header("Training & Model Artifacts", "Generate the saved pipeline used by the website.")
    model = load_model()
    model_status_box(model)

    st.subheader("Train the Final Pipeline")
    st.code("python train_model.py", language="bash")
    st.write(
        "The command loads `data/Fake.csv` and `data/True.csv`, cleans text, performs "
        "a stratified split, compares models with cross-validation, saves evaluation "
        "outputs, and writes `models/final_model.pkl`."
    )

    st.subheader("Optional Stacking Run")
    st.code("python train_model.py --include-stacking", language="bash")
    st.write(
        "Stacking is slower, but it is still a traditional machine learning ensemble."
    )

    artifacts = pd.DataFrame(
        [
            {"File": "models/final_model.pkl", "Status": "Found" if MODEL_PATH.exists() else "Missing"},
            {
                "File": "outputs/model_comparison.csv",
                "Status": "Found" if (OUTPUTS_DIR / "model_comparison.csv").exists() else "Missing",
            },
            {
                "File": "outputs/classification_report.txt",
                "Status": "Found" if (OUTPUTS_DIR / "classification_report.txt").exists() else "Missing",
            },
            {
                "File": "outputs/confusion_matrix.csv",
                "Status": "Found" if (OUTPUTS_DIR / "confusion_matrix.csv").exists() else "Missing",
            },
        ]
    )
    st.dataframe(artifacts, use_container_width=True, hide_index=True)


def show_results() -> None:
    page_header("Evaluation Dashboard", "Model comparison and final evaluation outputs.")
    comparison = read_csv_if_exists(OUTPUTS_DIR / "model_comparison.csv")
    if comparison is None:
        st.warning("No saved evaluation table found yet. Run `python train_model.py` first.")
        st.write("After training, this page will display model comparison results and final metrics.")
    else:
        st.subheader("Model Comparison")
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
        best = comparison.sort_values("CV_Macro_F1_Mean", ascending=False).iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", str(best["Model"]))
        col2.metric("CV Macro F1", f"{best['CV_Macro_F1_Mean']:.4f}")
        col3.metric("Test Macro F1", f"{best['Test_Macro_F1']:.4f}")

    matrix = read_csv_if_exists(OUTPUTS_DIR / "confusion_matrix.csv")
    if matrix is not None:
        st.subheader("Confusion Matrix")
        st.dataframe(matrix, use_container_width=True, hide_index=True)

    report_path = OUTPUTS_DIR / "classification_report.txt"
    if report_path.exists():
        st.subheader("Classification Report")
        st.code(report_path.read_text(encoding="utf-8"), language="text")

    bootstrap_path = OUTPUTS_DIR / "bootstrap_confidence_intervals.txt"
    if bootstrap_path.exists():
        st.subheader("Bootstrap Confidence Intervals")
        st.code(bootstrap_path.read_text(encoding="utf-8"), language="text")

    roc_path = OUTPUTS_DIR / "roc_curve.png"
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    image_cols = st.columns(2)
    if roc_path.exists():
        image_cols[0].image(str(roc_path), caption="ROC Curve", use_container_width=True)
    if cm_path.exists():
        image_cols[1].image(str(cm_path), caption="Confusion Matrix", use_container_width=True)

    st.info(
        "High same-dataset F1 should be interpreted carefully. Strong test performance on "
        "one dataset does not automatically prove real-world generalization."
    )


def show_prediction() -> None:
    page_header(
        "Interactive Prediction",
        "Paste a headline or article and classify it with the saved traditional ML pipeline.",
    )
    model = load_model()
    model_status_box(model)

    user_text = st.text_area(
        "News headline or article",
        height=220,
        placeholder="Paste a news headline or article here...",
    )

    col1, col2 = st.columns([1, 3])
    predict_clicked = col1.button("Predict", type="primary", use_container_width=True)
    col2.caption("The app cleans the input text before passing it into the saved TF-IDF + classifier pipeline.")

    with st.expander("Preview the cleaning rules"):
        st.write("Remove URLs, remove HTML tags, lowercase, remove non-letter characters, remove extra spaces.")
        if user_text:
            st.code(clean_text(user_text), language="text")

    if predict_clicked:
        if model is None:
            st.error("Prediction is unavailable until `models/final_model.pkl` is created.")
            st.code("python train_model.py", language="bash")
            return
        if not user_text.strip():
            st.warning("Please enter a headline or article first.")
            return

        cleaned = clean_text(user_text)
        prediction = int(model.predict([cleaned])[0])
        score = model_scores(model, [cleaned])
        label = LABEL_NAMES.get(prediction, str(prediction))

        if prediction == 0:
            st.error(f"Predicted label: {label}")
        else:
            st.success(f"Predicted label: {label}")

        if score is not None:
            value = float(score[0])
            if hasattr(model, "predict_proba"):
                confidence = value if prediction == 1 else 1 - value
                st.metric("Confidence", f"{confidence:.2%}")
                st.caption(f"Real-news probability: {value:.4f}")
            else:
                st.metric("Decision score", f"{value:.4f}")

        st.info(
            "This prediction is based on learned text patterns, not direct factual verification. "
            "A real article can still be misclassified if its wording resembles patterns learned "
            "from fake-news examples, or if the dataset contains source/style bias."
        )


def show_limitations() -> None:
    page_header("Limitations & Future Work", "How to interpret this project responsibly.")
    st.subheader("Limitations")
    limitations = [
        "This is a text classification system, not a real fact-checking system.",
        "It cannot directly verify factual correctness against external evidence.",
        "It may learn source-specific, topic-specific, or writing-style patterns.",
        "It may struggle with partially true, mixed, satire, or very short news text.",
        "Performance on the same dataset may overestimate real-world generalization.",
    ]
    for item in limitations:
        st.write(f"- {item}")

    st.subheader("Future Work")
    future = [
        "Evaluate on external datasets not used during model development.",
        "Use source-based or time-based train/test splits.",
        "Add more diverse news sources and publication periods.",
        "Try non-deep-learning text representations such as Word2Vec features.",
        "Add metadata features such as source credibility, author, or date.",
        "Optionally integrate a fact-checking API as a separate evidence layer.",
    ]
    for item in future:
        st.write(f"- {item}")


def main() -> None:
    st.sidebar.title("News Classification")
    st.sidebar.caption("Traditional machine learning project website")
    pages = {
        "Home": show_home,
        "Dataset Explorer": show_dataset,
        "Methodology": show_methodology,
        "Training & Artifacts": show_training,
        "Evaluation Dashboard": show_results,
        "Interactive Prediction": show_prediction,
        "Limitations & Future Work": show_limitations,
    }
    selected = st.sidebar.radio("Pages", list(pages.keys()))

    st.sidebar.divider()
    if MODEL_PATH.exists():
        st.sidebar.success("Model file found")
    else:
        st.sidebar.warning("Model file missing")
    st.sidebar.caption("Label convention: Fake News = 0, Real News = 1")

    pages[selected]()


if __name__ == "__main__":
    main()
