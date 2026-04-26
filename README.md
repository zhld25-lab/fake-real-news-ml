# Fake vs Real News Classification using Machine Learning

## Project Overview

Fake news detection has become an important problem in modern society due to the rapid spread of misinformation on the internet. Social media and online news platforms allow false information to spread quickly, which can influence public opinion and decision making.

This project develops a machine learning system that classifies news articles as **fake news or real news** based on their textual content. The system processes raw news data, applies text preprocessing, extracts features using **TF-IDF vectorization**, and trains machine learning models to perform binary classification.

The project focuses on classical machine learning methods combined with statistical evaluation techniques to build a reliable and interpretable fake news detection pipeline.

---

## Streamlit Web App

This repository includes a Streamlit web application for interactive fake news prediction.

The app includes these pages:

- Home
- Interactive Prediction
- Model Performance
- Confusion Matrix
- Classification Report
- Bootstrap Confidence Intervals
- Submission Information

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For public deployment, use Streamlit Community Cloud:

1. Go to `https://share.streamlit.io`.
2. Create a new app from this GitHub repository.
3. Select branch `main`.
4. Set the main file path to `app.py`.
5. Select Python `3.10`.
6. Deploy and share the generated `streamlit.app` URL.

Prediction requires `final_model.pkl` in the repository root. If it is missing, the app will still open and show a clear warning. Export the trained model from the notebook with:

```python
import joblib
joblib.dump(best_model, "final_model.pkl")
```

---

## Problem Description

The goal of this project is to automatically determine whether a news article is real or fake using machine learning.

This is formulated as a **binary text classification problem**.

Input  
News article text

Output  

1 = Real News  
0 = Fake News

The model learns patterns in textual content that help distinguish legitimate journalism from fabricated or misleading news articles.

---

## Dataset

The dataset used in this project consists of two files:

data/raw/Fake.csv  
data/raw/True.csv

Each dataset contains news articles collected from online news sources.

Typical attributes include:

- title
- text
- subject
- date

The datasets are merged and labeled as follows:

Fake News → label = 0  
Real News → label = 1

The title and text fields are combined to form a single text feature used for model training.

---

## Data Preprocessing

Before training the model, several preprocessing steps are applied to the raw text data.

Text preprocessing includes:

- Convert text to lowercase
- Remove URLs
- Remove punctuation and special characters
- Remove extra whitespace
- Remove empty rows

These steps help clean the text data and improve model performance.

---

## Feature Engineering

Text data is converted into numerical features using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

TF-IDF measures the importance of a word in a document relative to the entire corpus.

The vectorization process includes:

- Stopword removal
- N-gram representation
- Vocabulary size control

TF-IDF converts each document into a numerical feature vector that can be used by machine learning models.

---

## Baseline Solution

A baseline model is used as a reference point for evaluating machine learning performance.

The baseline approach in this project is **Naive Bayes**, which is commonly used for text classification tasks.

Naive Bayes is simple, efficient, and provides a good starting point for evaluating more advanced machine learning models.

---

## Machine Learning Model

The main model used in this project is **Logistic Regression**, which is widely used for binary classification problems.

The model is trained using TF-IDF features extracted from the news articles.

Logistic Regression is chosen because it performs well on high-dimensional text data and provides interpretable results.

---

## Statistical Model Components

To ensure reliable evaluation of model performance, statistical methods are applied.

### Cross Validation

A **5-fold cross validation** strategy is used during training.

This helps estimate the model's generalization performance and reduces the risk of overfitting.

### Evaluation Metrics

Model performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Among these metrics, **F1 Score** is used as the main evaluation metric because it balances precision and recall.

### Bootstrap Confidence Interval

Bootstrap resampling is used to estimate a **confidence interval for the F1 score**, providing a statistical measure of model reliability.

---

## Project Structure

```
fake-real-news-ml/

├── data/
│   └── raw/
│       ├── Fake.csv
│       └── True.csv

├── Fake News Detection.ipynb
├── README.md
├── bootstrap_confidence_intervals.txt
├── confusion_matrix_final_model.png
├── final_classification_report.txt
├── final_project_summary.txt
├── ranked_model_results.csv
├── roc_curve_final_model.png
└── top_terms_interpretability.txt
```


## How to Run the Project

1. Install required libraries

pip install pandas numpy scikit-learn matplotlib

2. Open the notebook

fake_real_news_model.ipynb

3. Run all cells to reproduce the training process and evaluation results.

---

## Results

The machine learning model is able to distinguish fake news from real news articles using textual features extracted through TF-IDF.

Model performance is evaluated using precision, recall, and F1 score.

The results demonstrate that classical machine learning methods combined with statistical evaluation techniques can effectively detect fake news.

---

## Authors

Zhenzhe Luo  
Xiaoxi Gao

Data Science Project  
Fake News Detection using Machine Learning
