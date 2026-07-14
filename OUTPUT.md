# 📰 Fake News Detection — Project Output Report

*A simple explanation of what this project does and what its results mean — written so anyone,
even without a coding background, can understand it.*

---

## What is this project?

This project teaches a computer to guess whether a news article is **FAKE** or **REAL**, just by
reading its text. It does this by studying thousands of news articles that are already labeled
as fake or real, learning the patterns and word choices that are common in each, and then using
that knowledge to make guesses on new articles it has never seen before.

Below is a walkthrough of every step the program went through, in plain language, along with the
actual screenshots of what it produced.

---

## Step 1: Loading the Data

The program starts by reading two files of news articles — one full of fake news articles, and
one full of real news articles — and combines them into a single table so it can study them
together.

<img width="426" height="121" alt="Screenshot 2026-07-14 171456" src="https://github.com/user-attachments/assets/8c687736-ba72-407d-9070-b325e8300415" />



**What this shows:** The fake news file had **23,481 articles**, and the real news file had
**21,417 articles**. Combined, that's a total of **44,898 news articles** for the computer to
learn from — a good, large amount of data.

---

## Step 2: Checking the Data is Clean

Before studying the data, it's important to check nothing is missing or broken — like checking
all the pages of a book are present before reading it.

<img width="155" height="134" alt="Screenshot 2026-07-14 171514" src="https://github.com/user-attachments/assets/859f3b45-3e21-41aa-8e58-059f6a5a85a3" />


<img width="182" height="128" alt="Screenshot 2026-07-14 171525" src="https://github.com/user-attachments/assets/cf6db28d-5284-4833-b848-c307296292cd" />


**What this shows:** Every single one of the 44,898 articles has complete information — there are
**zero missing values** anywhere. This also confirms the article counts: 23,481 fake articles
(label 0) and 21,417 real articles (label 1).

---

## Step 3: How Many Fake vs Real Articles Are There?

It helps to see this balance visually — if one category had far more articles than the other, the
computer might become biased toward guessing that category more often.

<img width="301" height="245" alt="Screenshot 2026-07-14 164833" src="https://github.com/user-attachments/assets/5d12efe3-558a-4875-a142-293a71f57cb7" />


**What this shows:** The dataset is fairly balanced — a bit more fake articles than real ones, but
not by a huge margin. This is good, because it means the computer gets a fair amount of practice
with both types.

---

## Step 4: How Long Are the Articles?

This chart shows how many words a typical article contains.

<img width="607" height="293" alt="Screenshot 2026-07-14 164855" src="https://github.com/user-attachments/assets/21c93a44-bd86-4cbb-9148-70fd46179ef7" />


**What this shows:** Most articles are on the shorter side — somewhere between 100 and 500 words
— with fewer and fewer articles as length increases. This is normal for news articles, which are
usually written to be concise.

---

## Step 5: Which Words Show Up the Most? (Word Clouds)

A "word cloud" is a picture where words that appear more often in the text are shown **bigger**.
This gives a quick visual sense of what each category of news tends to talk about.

<img width="553" height="266" alt="Screenshot 2026-07-14 164914" src="https://github.com/user-attachments/assets/f6a9442b-6846-4230-b73c-4b7678b1c369" />


<img width="429" height="222" alt="Screenshot 2026-07-14 164931" src="https://github.com/user-attachments/assets/13f7f3ab-0426-47b9-98fb-f1c64dfb7b9f" />


**What this shows:** Both fake and real articles talk a lot about similar big-picture topics
(politics, Trump, government), which makes sense since this dataset is mostly political news.
The real news word cloud shows more formal words like "Reuters" and "Washington" (a wire news
agency and where the US government is based), while the fake news word cloud leans more on
emotionally charged or informal words. This difference in *style*, even when discussing similar
topics, is part of what the computer learns to detect.

---

## Step 6: The Most Common Words, and Cleaning the Text

Before the computer can really learn from the text, the text needs to be "cleaned" — removing
things like punctuation, numbers, and very common filler words (like "the", "is", "and") that
don't carry much meaning. Each word is also reduced to its simplest form (for example, "running"
becomes "run").

<img width="590" height="185" alt="Screenshot 2026-07-14 171720" src="https://github.com/user-attachments/assets/dd4f5d79-07d3-4bc7-ac7d-e7d7ee044d65" />


**What this shows:**
- The top words confirm what the word clouds showed — words like "trump", "said", "president"
  dominate both categories.
- The **before/after cleaning example** shows this in action:
  - *Before:* "The Government announced 25 NEW policies in 2024!!!"
  - *After:* "government announced new policy"

  Notice how the cleaned version keeps only the meaningful words, in their simplest form, and
  removes punctuation, capitalization, and numbers. This makes it much easier for the computer to
  spot patterns.
- After cleaning, the data was split into two parts: **35,918 articles** to teach the computer
  (the "training set"), and **8,980 articles** to test how well it learned (the "testing set") —
  articles the computer never saw during training, used purely to check its performance fairly.

---

## Step 7: Turning Words Into Numbers, and Training the Models

Computers can't understand words directly — they need numbers. So each cleaned article is
converted into a set of numbers that represent which words it contains and how important each
word is (this technique is called **TF-IDF**). Once the text is in number form, the computer can
start learning patterns from it.

Three different learning methods (called "models") were trained separately, so their results
could be compared:

<img width="277" height="143" alt="Screenshot 2026-07-14 171731" src="https://github.com/user-attachments/assets/db42bdcb-5b3b-4f5f-8cd9-e9e1f3a9227e" />


**What this shows:** The text was converted into **3,000 number-features** per article, and all
three models — **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear SVM** —
finished training successfully.

---

## Step 8: How Well Did Each Model Do?

Now for the important part — checking how good each model actually is at telling fake news from
real news. A few simple measurements are used:

- **Accuracy** — out of all the test articles, what percentage did the model guess correctly?
- **Precision** — when the model says an article is FAKE (or REAL), how often is it actually
  right?
- **Recall** — out of all the truly FAKE (or REAL) articles, how many did the model successfully
  catch?
- **F1-Score** — a single balanced score that combines Precision and Recall together.
- **Confusion Matrix** — a simple table showing exactly how many articles were correctly guessed,
  and how many were mixed up.

### Model 1: Logistic Regression

<img width="293" height="184" alt="Screenshot 2026-07-14 171742" src="https://github.com/user-attachments/assets/0b5b8e87-d259-4cf4-ab22-f766863035ce" />


<img width="235" height="200" alt="Screenshot 2026-07-14 171312" src="https://github.com/user-attachments/assets/b4bced30-7311-4520-8b47-2d15fc180abe" />


**What this shows:** This model got **98.72% accuracy** — meaning out of every 100 articles, it
correctly guessed about 99 of them. Looking at the confusion matrix: out of 4,696 truly fake
articles, it correctly caught 4,624 of them (and only mistakenly called 72 of them real). Out of
4,284 truly real articles, it correctly identified 4,241 (only 43 mistaken as fake). That's a
very strong result.

### Model 2: Multinomial Naive Bayes

<img width="263" height="197" alt="Screenshot 2026-07-14 171753" src="https://github.com/user-attachments/assets/1bd7e316-b6af-4a11-9303-e193f3b278d7" />


<img width="233" height="196" alt="Screenshot 2026-07-14 171326" src="https://github.com/user-attachments/assets/b9db325e-4a43-409d-a3f0-8b30becc5f75" />


**What this shows:** This model got **93.01% accuracy** — still good, but noticeably lower than
the other two. Its confusion matrix shows more mix-ups: 296 fake articles were wrongly called
real, and 332 real articles were wrongly called fake. This model is the fastest and simplest of
the three, but here it traded off a bit of accuracy for that simplicity.

### Model 3: Linear SVM

<img width="256" height="191" alt="Screenshot 2026-07-14 171802" src="https://github.com/user-attachments/assets/41d079bf-33d5-4a29-914c-5ed8de8f07b0" />


<img width="233" height="193" alt="Screenshot 2026-07-14 171342" src="https://github.com/user-attachments/assets/81be2ecb-f937-46f1-9bba-7953a5be30f9" />


**What this shows:** This model performed the best, with **99.34% accuracy**. Its confusion
matrix shows only 31 fake articles wrongly called real, and just 28 real articles wrongly called
fake — out of nearly 9,000 test articles, that's an extremely small number of mistakes.

---

## Step 9: Comparing All Three Models Side by Side

<img width="569" height="298" alt="Screenshot 2026-07-14 171400" src="https://github.com/user-attachments/assets/b0b569de-b241-4525-a29d-e7b33994212a" />


<img width="319" height="106" alt="Screenshot 2026-07-14 171814" src="https://github.com/user-attachments/assets/ee9b32d8-fabf-43a0-b225-40027ab2b301" />


**What this shows:** All three models did well, but **Linear SVM came out on top** with the
highest accuracy (99.34%), narrowly beating Logistic Regression (98.72%), with Multinomial Naive
Bayes trailing behind at 93.01%. Because Linear SVM performed best, the program automatically
selected it as the final model and saved it to disk — so it's ready to be reused any time without
having to retrain from scratch.

---

## Step 10: Putting the Model to the Test

Finally, the saved model was tested on a few brand-new headlines it had never seen before, to see
how it performs in a real, practical scenario. Along with each guess, it also gives a
**confidence score** — how sure it is about its answer.

<img width="349" height="219" alt="Screenshot 2026-07-14 171837" src="https://github.com/user-attachments/assets/13cfa2e1-a83c-461b-9a6f-d9ff9690a4ae" />


**What this shows:**

| Headline | Prediction | Confidence |
|---|---|---|
| "Scientists at NASA confirm successful landing of new rover on Mars." | FAKE | 79.8% |
| "BREAKING: Aliens spotted controlling world governments through secret towers." | FAKE | 94.5% |
| "The Reserve Bank of India raised its key interest rate on Wednesday." | REAL | 59.73% |
| "Miracle fruit found in Amazon rainforest cures all diseases overnight!" | FAKE | 69.66% |

The obviously fake/sensational headlines (aliens, miracle fruit) were correctly flagged as FAKE
with reasonably high confidence. Interestingly, the genuine NASA headline was also marked FAKE —
this happens because the model only learned from **political news** articles, and has never seen
science/space headlines before, so it doesn't have a strong basis to judge topics outside what it
was trained on. This is an honest and expected limitation, not a bug — it highlights that the
model is good at recognizing *patterns it has seen before*, not at truly understanding facts.

---

## Overall Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 98.72% | 98.33% | 98.99% | 98.66% |
| Multinomial Naive Bayes | 93.01% | 93.02% | 92.25% | 92.64% |
| **Linear SVM (Best Model)** | **99.34%** | **99.28%** | **99.35%** | **99.31%** |

The final saved model — **Linear SVM** — correctly classifies about **99 out of every 100**
news articles from this dataset, making it a highly effective tool for detecting fake news
patterns within the political news domain it was trained on.

---

*End of Report.*
