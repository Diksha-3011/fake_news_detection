# 📰 Fake News Detection using Machine Learning and Natural Language Processing



---

## 📌 Project Abstract

Fake news spreads quickly on digital platforms and can mislead public opinion, so there is a
real need for automated tools that can flag unreliable news content. This project builds a
simple **Fake News Detection System** using classical **Machine Learning (ML)** and **Natural
Language Processing (NLP)** techniques.

News articles (title + text) are cleaned using standard NLP preprocessing steps — lowercasing,
punctuation and number removal, stopword removal, tokenization, and lemmatization — and then
converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.
Three classification models — **Logistic Regression**, **Multinomial Naive Bayes**, and
**Linear Support Vector Machine (SVM)** — are trained and compared using Accuracy, Precision,
Recall, F1-Score, and Confusion Matrices. The best-performing model is saved using **Joblib**
and used in a simple function that predicts whether a given news headline or article is
**FAKE** or **REAL**, along with a confidence percentage.

---

## 🎯 Project Objectives

1. Explore and understand a real-world fake/real news dataset (EDA).
2. Clean and prepare text data using NLTK-based NLP preprocessing.
3. Convert text into numerical features using TF-IDF.
4. Train and evaluate three different ML classifiers.
5. Compare model performance using standard classification metrics.
6. Save the best model so it can be reused without retraining.
7. Build a simple function to classify any new headline as FAKE or REAL.

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python 3 |
| Data Handling | Pandas, NumPy |
| NLP | NLTK (stopwords, tokenizer, WordNet Lemmatizer) |
| Feature Engineering | Scikit-learn `TfidfVectorizer` |
| Machine Learning | Scikit-learn (Logistic Regression, Multinomial Naive Bayes, Linear SVM) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Model Persistence | Joblib |
| Environment | VS Code (or Jupyter Notebook / Kaggle Kernel) |

---

## 💻 Software Requirements

- Python 3.9+ (3.11 or 3.12 recommended)
- VS Code with the Python extension, or Jupyter Notebook
- Libraries: `pandas`, `numpy`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`, `wordcloud`, `joblib`
- Internet access for the one-time NLTK data download

## 🖥️ Hardware Requirements

- Minimum 4 GB RAM (8 GB recommended)
- Dual-core CPU or better (no GPU needed — these are classical ML models, not deep learning)
- ~500 MB free disk space for the dataset and saved model files

---

## 🔄 Workflow Diagram


*(Data flows top-to-bottom: raw dataset → labeling → EDA → NLP preprocessing (lowercasing,
punctuation/number removal, stopword removal, tokenization, lemmatization) → cleaned corpus →
TF-IDF feature extraction → train/test split → three models (Logistic Regression, Multinomial
Naive Bayes, Linear SVM) → evaluation & comparison → best model selection → save with Joblib →
FAKE/REAL prediction with confidence score.)*

---

## 📂 Dataset

This project uses the **"Fake and Real News Dataset"** by Clément Bisaillon, available on Kaggle:

> https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It contains **two CSV files**:
- `Fake.csv` — fake news articles (label = 0)
- `True.csv` — real news articles (label = 1)

Both files have the columns: `title`, `text`, `subject`, `date`.

### Setup

1. Download the dataset from the link above and unzip it.
2. Create a folder named `data` next to `fake_news_detection.py`.
3. Place both `Fake.csv` and `True.csv` inside that `data` folder:

```
news_ml/
├── fake_news_detection.py
├── README.md
├── workflow_diagram.png
└── OUTPUT.md

```

---

## ▶️ How to Run

1. Open the project folder in VS Code.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1        (Windows PowerShell)
   ```
3. Install the required libraries:
   ```
   pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud joblib
   ```
4. Run the script:
   ```
   python fake_news_detection.py
   ```
5. The first run downloads some NLTK data automatically (needs internet, one-time only).
6. Chart windows will pop up one at a time — close each one to let the script continue.
7. Cleaning the text and training the models can take a few minutes on the full dataset —
   this is normal, just wait for each "Step ... done" message.
8. At the end, you can type in your own news headline to test the model, or press Enter to skip.

---

## 🧠 What the Code Does (Step by Step)

The script (`fake_news_detection.py`) is organized into 10 clear steps:

1. **Import libraries** — loads all the Python packages needed.
2. **Download NLTK data** — stopwords list, tokenizer, and lemmatizer (one-time).
3. **Load the dataset** — reads `Fake.csv` and `True.csv`, labels them (0 = FAKE, 1 = REAL),
   and combines them into one table.
4. **Explore the data (EDA)** — checks the dataset shape, missing values, class balance,
   article length, word clouds, and the most frequent words in each class.
5. **Clean the text** — lowercases, removes links/numbers/punctuation, removes stopwords,
   and reduces each word to its base form (lemmatization) using NLTK.
6. **TF-IDF vectorization** — converts the cleaned text into numeric features the models
   can learn from.
7. **Train 3 models** — Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
8. **Evaluate the models** — prints Accuracy, Precision, Recall, F1-Score, and a confusion
   matrix for each model, plus a bar chart comparing them.
9. **Pick and save the best model** — automatically chooses the model with the highest
   accuracy and saves it (plus the TF-IDF vectorizer) using Joblib as `.pkl` files.
10. **Predict new headlines** — a `predict_news()` function that takes any text and returns
    FAKE or REAL along with a confidence percentage; also lets you type in your own headline.

---

## 📊 Results

Exact numbers will depend on your run, but you should expect fairly high accuracy
(often 90%+) on this dataset, since the fake and real articles differ quite noticeably in
writing style and vocabulary. The comparison table and bar chart printed in Step 8 show the
exact Accuracy, Precision, Recall, and F1-Score for all three models, and Step 9 prints which
model was automatically selected as the best one.

---

## ✅ Conclusion

This project shows how basic NLP preprocessing combined with TF-IDF and simple ML classifiers
can effectively separate fake news from real news. Logistic Regression, Naive Bayes, and Linear
SVM all perform well on this dataset, and the best one is automatically selected, saved, and
made available for real-time predictions on new headlines.

## 🚀 Future Scope

- Try deep learning models (LSTM, BERT) to capture more context than TF-IDF can.
- Use a more diverse dataset covering more topics than just politics.
- Deploy the model as a simple web app so anyone can check a headline.
- Add explainability (e.g. highlighting which words influenced the prediction).

## ⚠️ Limitations

- The model only learns from the patterns in this specific dataset — it may not generalize
  well to news on topics that aren't well represented here (e.g. sports, entertainment).
- It checks writing style and word patterns, not actual facts — it does not "know" if an
  event really happened.
- A well-written fake article or a poorly-written real one could still fool the model.

## 📚 References

1. Bisaillon, C. — *Fake and Real News Dataset*, Kaggle.
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of
   Machine Learning Research, 12, 2825–2830.
3. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*.
   O'Reilly Media. (https://www.nltk.org/)

---



*End of README — Fake News Detection using Machine Learning and Natural Language Processing.*
