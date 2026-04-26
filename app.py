import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "final_model.pkl"
ROC_CURVE_PATH = APP_DIR / "roc_curve_final_model.png"
CONFUSION_MATRIX_IMAGE_PATH = APP_DIR / "confusion_matrix_final_model.png"

MODEL_EXPORT_SNIPPET = """import joblib
joblib.dump(best_model, "final_model.pkl")"""

PERFORMANCE = {
    "Final Model": "Stacking",
    "Accuracy": 0.9994,
    "Macro F1-score": 0.9994,
    "Wrong Predictions": "5 / 8935",
}

CONFUSION_MATRIX = pd.DataFrame(
    [[4690, 4], [1, 4240]],
    index=["Actual Fake", "Actual True"],
    columns=["Predicted Fake", "Predicted True"],
)

CLASSIFICATION_REPORT = pd.DataFrame(
    [
        {"Class": "Fake", "Precision": 0.9998, "Recall": 0.9991, "F1-score": 0.9995, "Support": 4694},
        {"Class": "True", "Precision": 0.9991, "Recall": 0.9998, "F1-score": 0.9994, "Support": 4241},
        {"Class": "Accuracy", "Precision": np.nan, "Recall": np.nan, "F1-score": 0.9994, "Support": 8935},
        {"Class": "Macro avg", "Precision": 0.9994, "Recall": 0.9995, "F1-score": 0.9994, "Support": 8935},
        {"Class": "Weighted avg", "Precision": 0.9994, "Recall": 0.9994, "F1-score": 0.9994, "Support": 8935},
    ]
)

BOOTSTRAP_INTERVALS = pd.DataFrame(
    [
        {"Metric": "Accuracy", "Mean": 0.9994, "95% CI Lower": 0.9989, "95% CI Upper": 0.9999},
        {"Metric": "F1 Score", "Mean": 0.9994, "95% CI Lower": 0.9988, "95% CI Upper": 0.9999},
    ]
)

PAGES = [
    "Home",
    "Interactive Prediction",
    "Model Performance",
    "Confusion Matrix",
    "Classification Report",
    "Bootstrap Confidence Intervals",
    "Submission Information",
]


st.set_page_config(
    page_title="Fake vs Real News Detector",
    page_icon="NEWS",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2.5rem;
                max-width: 1180px;
            }
            h1, h2, h3 {
                letter-spacing: 0;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(49, 51, 63, 0.12);
            }
            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid rgba(49, 51, 63, 0.12);
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            }
            div.stButton > button {
                width: 100%;
                border-radius: 8px;
                font-weight: 700;
                padding: 0.65rem 1rem;
            }
            .status-panel {
                border: 1px solid rgba(49, 51, 63, 0.12);
                border-radius: 8px;
                padding: 1rem;
                background: #f8fafc;
            }
            .result-panel {
                border: 1px solid rgba(49, 51, 63, 0.12);
                border-radius: 8px;
                padding: 1.25rem;
                background: #ffffff;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_trained_model() -> tuple[Any | None, str | None]:
    if not MODEL_PATH.exists():
        return None, "missing"

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f"load_error: {exc}"


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_confusion_matrix() -> pd.DataFrame:
    return CONFUSION_MATRIX.apply(lambda column: column.map(lambda value: f"{int(value):,}"))


def format_classification_report() -> pd.DataFrame:
    report = CLASSIFICATION_REPORT.copy()
    for column in ["Precision", "Recall", "F1-score"]:
        report[column] = report[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    report["Support"] = report["Support"].map(lambda value: f"{int(value):,}")
    return report


def format_bootstrap_intervals() -> pd.DataFrame:
    intervals = BOOTSTRAP_INTERVALS.copy()
    for column in ["Mean", "95% CI Lower", "95% CI Upper"]:
        intervals[column] = intervals[column].map(lambda value: f"{value:.4f}")
    return intervals


def label_to_text(label: Any) -> str:
    normalized = str(label).strip().lower()
    if normalized in {"0", "fake", "fake news"}:
        return "Fake News"
    if normalized in {"1", "true", "true news", "real", "real news"}:
        return "True News"
    return "True News" if bool(label) else "Fake News"


def prediction_confidence(model: Any, cleaned_text: str, predicted_label: Any) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probabilities = model.predict_proba([cleaned_text])[0]
    except Exception:
        return None

    classes = getattr(model, "classes_", None)
    if classes is not None:
        for index, class_label in enumerate(classes):
            if str(class_label) == str(predicted_label):
                return float(probabilities[index])

    try:
        label_index = int(predicted_label)
        if 0 <= label_index < len(probabilities):
            return float(probabilities[label_index])
    except (TypeError, ValueError):
        pass

    return float(np.max(probabilities))


def show_model_status(model: Any | None, model_error: str | None) -> None:
    if model is not None:
        st.success("final_model.pkl loaded successfully. The prediction page is ready.")
        return

    if model_error == "missing":
        st.warning(
            "final_model.pkl was not found. Export the trained model from the notebook and place it "
            "next to app.py before running predictions."
        )
        st.code(MODEL_EXPORT_SNIPPET, language="python")
        return

    st.error("final_model.pkl exists, but Streamlit could not load it.")
    st.caption(model_error.replace("load_error: ", "") if model_error else "Unknown model loading error.")


def render_header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def render_home(model: Any | None, model_error: str | None) -> None:
    render_header(
        "Fake vs Real News Detector",
        "Interactive Streamlit application for classifying news text with a trained machine learning model.",
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Final Model", PERFORMANCE["Final Model"])
    metric_cols[1].metric("Accuracy", f"{PERFORMANCE['Accuracy']:.4f}")
    metric_cols[2].metric("Macro F1-score", f"{PERFORMANCE['Macro F1-score']:.4f}")
    metric_cols[3].metric("Wrong Predictions", PERFORMANCE["Wrong Predictions"])

    st.divider()

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Application Overview")
        st.write(
            "Use the sidebar to review model performance, inspect evaluation tables, and run an "
            "interactive fake news prediction on a headline or full article."
        )
        with st.expander("Text preprocessing used before prediction", expanded=True):
            st.write(
                "The app removes URLs and HTML tags, lowercases text, removes non-letter characters, "
                "and normalizes extra whitespace before passing the cleaned text to the model."
            )
    with right:
        st.subheader("Model Status")
        show_model_status(model, model_error)


def render_prediction(model: Any | None, model_error: str | None) -> None:
    render_header(
        "Interactive Prediction",
        "Enter a headline or article, then classify it as Fake News or True News.",
    )

    show_model_status(model, model_error)

    user_text = st.text_area(
        "News headline or article",
        height=220,
        placeholder="Paste a news headline or article text here...",
    )

    predict_clicked = st.button("Predict", type="primary")

    if not predict_clicked:
        with st.expander("Preview the cleaning rules"):
            st.write("URLs, HTML tags, punctuation, digits, and extra spaces are removed before prediction.")
        return

    if model is None:
        st.warning("Prediction is unavailable until final_model.pkl is added and loaded successfully.")
        return

    cleaned = clean_text(user_text)
    if not cleaned:
        st.warning("Please enter text with at least a few alphabetic words before predicting.")
        return

    try:
        predicted_label = model.predict([cleaned])[0]
        result = label_to_text(predicted_label)
        confidence = prediction_confidence(model, cleaned, predicted_label)
    except Exception as exc:
        st.error("The model could not make a prediction for this input.")
        st.caption(str(exc))
        return

    result_cols = st.columns([1, 1])
    with result_cols[0]:
        if result == "True News":
            st.success(result)
        else:
            st.error(result)
    with result_cols[1]:
        if confidence is not None:
            st.metric("Confidence", f"{confidence:.2%}")
        else:
            st.info("This model does not expose predict_proba, so no confidence score is available.")

    with st.expander("Cleaned text sent to the model", expanded=False):
        st.write(cleaned)


def render_performance() -> None:
    render_header(
        "Model Performance",
        "Summary metrics for the final trained stacking model.",
    )

    cols = st.columns(4)
    cols[0].metric("Final Model", PERFORMANCE["Final Model"])
    cols[1].metric("Accuracy", f"{PERFORMANCE['Accuracy']:.4f}")
    cols[2].metric("Macro F1-score", f"{PERFORMANCE['Macro F1-score']:.4f}")
    cols[3].metric("Wrong Predictions", PERFORMANCE["Wrong Predictions"])

    performance_table = pd.DataFrame(
        [
            {"Metric": "Final Model", "Value": PERFORMANCE["Final Model"]},
            {"Metric": "Accuracy", "Value": f"{PERFORMANCE['Accuracy']:.4f}"},
            {"Metric": "Macro F1-score", "Value": f"{PERFORMANCE['Macro F1-score']:.4f}"},
            {"Metric": "Wrong Predictions", "Value": PERFORMANCE["Wrong Predictions"]},
        ]
    )
    st.subheader("Performance Table")
    st.dataframe(performance_table, hide_index=True, width="stretch")

    st.subheader("ROC Curve")
    if ROC_CURVE_PATH.exists():
        st.image(str(ROC_CURVE_PATH), caption="ROC Curve - Final Model", width="stretch")
    else:
        st.info("roc_curve_final_model.png was not found, so the ROC curve image is hidden.")


def render_confusion_matrix() -> None:
    render_header(
        "Confusion Matrix",
        "Prediction counts split by actual and predicted class.",
    )

    correct = int(CONFUSION_MATRIX.loc["Actual Fake", "Predicted Fake"] + CONFUSION_MATRIX.loc["Actual True", "Predicted True"])
    total = int(CONFUSION_MATRIX.to_numpy().sum())
    mistakes = total - correct

    cols = st.columns(3)
    cols[0].metric("Correct Predictions", f"{correct:,}")
    cols[1].metric("Wrong Predictions", f"{mistakes:,}")
    cols[2].metric("Total Samples", f"{total:,}")

    st.subheader("Matrix")
    st.dataframe(format_confusion_matrix(), width="stretch")

    if CONFUSION_MATRIX_IMAGE_PATH.exists():
        with st.expander("View confusion matrix image"):
            st.image(
                str(CONFUSION_MATRIX_IMAGE_PATH),
                caption="Confusion Matrix - Final Model",
                width="stretch",
            )

    with st.expander("Raw values"):
        st.write("Actual Fake predicted Fake: 4690")
        st.write("Actual Fake predicted True: 4")
        st.write("Actual True predicted Fake: 1")
        st.write("Actual True predicted True: 4240")


def render_classification_report() -> None:
    render_header(
        "Classification Report",
        "Precision, recall, F1-score, and support by class.",
    )

    st.dataframe(format_classification_report(), hide_index=True, width="stretch")

    with st.expander("Report notes"):
        st.write("Label convention used by the app: 0 = Fake News, 1 = True News.")
        st.write("Accuracy: 0.9994 on 8935 held-out examples.")


def render_bootstrap_intervals() -> None:
    render_header(
        "Bootstrap Confidence Intervals",
        "Estimated uncertainty for the final model metrics.",
    )

    cols = st.columns(2)
    cols[0].metric("Accuracy Mean", "0.9994", "95% CI 0.9989 to 0.9999")
    cols[1].metric("F1 Score Mean", "0.9994", "95% CI 0.9988 to 0.9999")

    st.subheader("Confidence Interval Table")
    st.dataframe(format_bootstrap_intervals(), hide_index=True, width="stretch")

    chart_data = BOOTSTRAP_INTERVALS.set_index("Metric")[["Mean", "95% CI Lower", "95% CI Upper"]]
    st.bar_chart(chart_data)


def render_submission_info(model: Any | None, model_error: str | None) -> None:
    render_header(
        "Submission Information",
        "How to run the app and prepare the final trained model artifact.",
    )

    st.subheader("Run Locally")
    st.code("streamlit run app.py", language="bash")

    st.subheader("Share on the Same Wi-Fi")
    st.write(
        "For a short demo on the same network, run Streamlit on all network interfaces, then share "
        "your computer's local IPv4 address with the port."
    )
    st.code("streamlit run app.py --server.address 0.0.0.0 --server.port 8501", language="bash")
    st.code("http://YOUR-IP-ADDRESS:8501", language="text")

    st.subheader("Deploy for Public Access")
    st.write(
        "For a link that works on other people's computers anywhere, deploy the project to Streamlit "
        "Community Cloud from GitHub and use app.py as the entrypoint."
    )
    st.code("https://share.streamlit.io", language="text")
    with st.expander("Public deployment checklist", expanded=True):
        st.write("Push app.py, requirements.txt, README.md, and .streamlit/config.toml to GitHub.")
        st.write("Add final_model.pkl to the repository if you want online predictions to work.")
        st.write("Create a new Streamlit Community Cloud app from the repository.")
        st.write("Set the main file path to app.py.")
        st.write("Share the generated streamlit.app URL.")

    st.subheader("Required Model File")
    show_model_status(model, model_error)

    with st.expander("Export final_model.pkl from the notebook", expanded=True):
        st.write("After training and selecting the best model, run this in the notebook:")
        st.code(MODEL_EXPORT_SNIPPET, language="python")

    with st.expander("Included app files", expanded=True):
        st.write("app.py")
        st.write("requirements.txt")
        st.write("README.md")
        st.write(".streamlit/config.toml")


def render_sidebar(model: Any | None) -> str:
    st.sidebar.title("News Detector")
    page = st.sidebar.radio("Pages", PAGES)
    st.sidebar.divider()
    st.sidebar.caption("Model file")
    if model is not None:
        st.sidebar.success("final_model.pkl loaded")
    else:
        st.sidebar.warning("final_model.pkl missing or unavailable")
    st.sidebar.caption("Label convention: 0 = Fake News, 1 = True News")
    return page


def main() -> None:
    apply_theme()
    model, model_error = load_trained_model()
    page = render_sidebar(model)

    if page == "Home":
        render_home(model, model_error)
    elif page == "Interactive Prediction":
        render_prediction(model, model_error)
    elif page == "Model Performance":
        render_performance()
    elif page == "Confusion Matrix":
        render_confusion_matrix()
    elif page == "Classification Report":
        render_classification_report()
    elif page == "Bootstrap Confidence Intervals":
        render_bootstrap_intervals()
    elif page == "Submission Information":
        render_submission_info(model, model_error)


if __name__ == "__main__":
    main()
