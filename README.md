# 📰 Fake News Detection using Machine Learning and Natural Language Processing

### A Final Year Major Project

---

## 📌 Project Abstract

The unchecked spread of misinformation through digital news platforms and social media has become
one of the most pressing challenges of the information age. Fake news can influence public opinion,
manipulate elections, incite panic, and erode trust in legitimate journalism. Manual fact-checking
is slow, expensive, and cannot keep pace with the volume of content generated every day.

This project presents an automated **Fake News Detection System** built using classical **Machine
Learning (ML)** algorithms combined with **Natural Language Processing (NLP)** techniques. Textual
news content (title + article body) is cleaned and normalized using standard NLP preprocessing
(lowercasing, punctuation/number removal, stopword removal, tokenization, and lemmatization), and
then converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**
vectorization. Three supervised classification models — **Logistic Regression**, **Multinomial Naive
Bayes**, and **Linear Support Vector Machine (SVM)** — are trained and rigorously compared using
Accuracy, Precision, Recall, F1-Score, and Confusion Matrices. The best-performing model is
persisted using **Joblib** and wrapped in a reusable prediction function that classifies any
user-supplied news headline or article as **FAKE** or **REAL**, along with a confidence score.

---

## 🎯 Project Objectives

1. To perform an in-depth Exploratory Data Analysis (EDA) of a real-world fake/real news dataset.
2. To design and implement a robust NLP text-preprocessing pipeline using NLTK.
3. To engineer meaningful numerical features from raw text using TF-IDF vectorization.
4. To train and evaluate multiple supervised ML classifiers for binary text classification.
5. To compare model performance objectively using standard classification metrics.
6. To select, persist, and deploy the best-performing model for real-time inference.
7. To build a simple, reusable interface that predicts whether a given news article is fake or real.

---

## 🛠️ Technologies Used

| Category             | Tools / Libraries |
|----------------------|-------------------|
| Programming Language | Python 3          |
| Data Handling        | Pandas, NumPy     |
| NLP                  | NLTK (stopwords, tokenizer, WordNet Lemmatizer) |
| Feature Engineering  | Scikit-learn `TfidfVectorizer` |
| Machine Learning     | Scikit-learn (Logistic Regression, Multinomial Naive Bayes, Linear SVM) |
| Visualization        | Matplotlib, Seaborn, WordCloud |
| Model Persistence    | Joblib |
| Environment          | vs code (Jupyter) |

---

## 💻 Software Requirements

- Python 3.9+
- Jupyter Notebook / Kaggle Kernel environment
- Libraries: `pandas`, `numpy`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`, `wordcloud`, `joblib`
- Internet access (Kaggle environment) for the one-time NLTK corpus download

## 🖥️ Hardware Requirements

- Minimum 4 GB RAM (8 GB recommended)
- Dual-core CPU or better (GPU is **not** required — classical ML models only)
- ~500 MB free disk space for dataset, model, and vectorizer artifacts

---


## 📂 Dataset

This notebook uses the **"Fake News"** dataset originally released for a Kaggle competition,
one of the most widely used beginner-friendly datasets for this task since it ships as a
**single labeled CSV file** (no manual merging of separate fake/real files required):

> https://www.kaggle.com/c/fake-news/data

It contains one file, `train.csv`, with the columns:
- `id` — unique identifier for each article
- `title` — the article headline
- `author` — the author of the article
- `text` — the full article body (may be incomplete for some rows)
- `label` — **1 = unreliable / fake news**, **0 = reliable / real news**

~20,800 labeled articles in total. Internally, this notebook re-maps the label to
**0 = FAKE, 1 = REAL** for consistency across all sections (EDA, metrics, confusion matrices,
and the final prediction function).

> **Before running:** click **Add Data** in the Kaggle notebook editor and add the dataset above
> (search "fake-news" and look for the competition dataset, or any mirrored copy that provides
> `train.csv` with the same columns). The loading cell below will automatically locate `train.csv`
> anywhere under `/kaggle/input/`, so no path editing is required.
