# Fake vs Real News Classification using Machine Learning

This repository is the official clean version of a course project for fake vs real news classification. It contains a traditional machine learning pipeline, command-line training and prediction scripts, and an interactive Streamlit website for project review.

## Project Type

- Task: binary text classification
- Fake news label: `0`
- Real news label: `1`
- Methods: traditional machine learning only
- Deep learning is not used: no BERT, Transformer, LSTM, CNN, TensorFlow, or PyTorch

## Project Structure

```text
fake-real-news-ml/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
├── outputs/
└── src/
    ├── data_utils.py
    ├── text_preprocessing.py
    ├── model_training.py
    ├── evaluation.py
    └── visualization.py
```

## Dataset

The official dataset files for this version are:

- `data/Fake.csv`
- `data/True.csv`

The loader standardizes the data into a consistent format with `title`, `text`, `subject`, `date`, `label`, `label_name`, and `source_file`.

## Pipeline

The main modeling pipeline is:

```text
load data -> clean text -> train/test split -> TF-IDF vectorization -> model comparison -> final evaluation -> bootstrap confidence intervals
```

TF-IDF is kept inside the scikit-learn `Pipeline`. This matters because cross-validation fits TF-IDF only on each training fold, which helps avoid data leakage.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

This command trains traditional machine learning models, compares them with cross-validation Macro F1, evaluates the selected model on the test set, and saves:

- `models/final_model.pkl`
- `outputs/model_comparison.csv`
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.csv`
- `outputs/bootstrap_confidence_intervals.txt`
- optional evaluation images

This clean version includes a saved `models/final_model.pkl` so the website prediction page can work after deployment. Re-run `python train_model.py` if you want to regenerate the model from the CSV data.

To also train the stacking ensemble:

```bash
python train_model.py --include-stacking
```

## Predict from Command Line

```bash
python predict.py --text "Your news headline or article text here"
```

The prediction uses the saved `models/final_model.pkl` pipeline.

## Run the Interactive Website

```bash
streamlit run app.py
```

The website includes:

- project overview
- dataset explorer
- methodology explanation
- training and artifact status
- evaluation dashboard
- interactive prediction page
- limitations and future work

If `models/final_model.pkl` is missing, the website still opens and clearly explains how to train and save the model.

## Evaluation Approach

The project reports:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Classification report
- Confusion matrix
- ROC curve and AUC when model scores are available
- Bootstrap confidence intervals for final metrics

High same-dataset performance should be interpreted carefully. Strong results on one dataset do not automatically prove real-world generalization. The model may learn topic, source, or writing-style patterns rather than factual correctness.

## Limitations

This project is a text-pattern classifier, not a factual verification system. It cannot directly check whether claims are true. It may struggle with partially true news, mixed claims, satire, short headlines, and articles from sources that differ from the training data.

Recommended future work includes external dataset testing, source-based or time-based splits, more diverse datasets, non-deep-learning embeddings such as Word2Vec features, and optional fact-checking API integration as a separate evidence layer.

## Removed Outdated Files

The previous repository version included old notebooks, old result images, old text reports, duplicated raw data folders, and temporary outputs. Those files were removed because this version reorganizes the repository into a clean submission structure with reproducible scripts and a complete interactive website.
