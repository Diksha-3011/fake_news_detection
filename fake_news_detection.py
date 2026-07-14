#  Import all the libraries we need


import os
import re
import string
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import joblib

RANDOM_STATE = 42

print("Step 1 done: libraries imported.")

# NLTK data 

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   # needed by newer nltk versions
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# list of common English words we will remove later, like "the", "is", "and"
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

print("Step 2 done: NLTK data ready.")


# Load the dataset


# read data
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

print("Fake.csv shape:", fake_df.shape)
print("True.csv shape:", true_df.shape)

# add a label column: 0 means FAKE news, 1 means REAL news
fake_df['label'] = 0
true_df['label'] = 1

# combine 
df = pd.concat([fake_df, true_df], axis=0)
df = df.reset_index(drop=True)

# fill any missing 
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

# combine the title and the text into one column called "content"
df['content'] = df['title'] + ' ' + df['text']


df = df.sample(frac=1.0, random_state=RANDOM_STATE)
df = df.reset_index(drop=True)

print("Step 3 done: dataset loaded. Shape:", df.shape)
print(df.head())


# Explore the dataset (EDA)


print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Class Distribution ---")
print(df['label'].value_counts())


label_names = {0: 'FAKE', 1: 'REAL'}
plt.figure(figsize=(6, 5))
sns.countplot(x=df['label'].map(label_names))
plt.title('Number of FAKE vs REAL Articles')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


df['word_count'] = df['content'].apply(lambda text: len(str(text).split()))

plt.figure(figsize=(8, 5))
sns.histplot(df['word_count'], bins=50)
plt.xlim(0, 2000)
plt.title('Distribution of Article Length (Word Count)')
plt.xlabel('Number of Words')
plt.ylabel('Number of Articles')
plt.show()

# word clouds for fake and real news
from wordcloud import WordCloud

fake_rows = df[df['label'] == 0]
real_rows = df[df['label'] == 1]

fake_sample_size = min(2000, len(fake_rows))
real_sample_size = min(2000, len(real_rows))

fake_news_text = ' '.join(fake_rows['content'].sample(fake_sample_size, random_state=RANDOM_STATE))
real_news_text = ' '.join(real_rows['content'].sample(real_sample_size, random_state=RANDOM_STATE))

plt.figure(figsize=(10, 6))
wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_news_text)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in FAKE News')
plt.show()

plt.figure(figsize=(10, 6))
wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_news_text)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in REAL News')
plt.show()

# most frequent words 
def get_most_common_words(text_list, how_many=20):
    all_words = []
    for text in text_list:
        words = str(text).lower().split()
        for word in words:
            word = word.strip(string.punctuation)
            if word != '' and word not in stop_words:
                all_words.append(word)
    return Counter(all_words).most_common(how_many)

top_fake_words = get_most_common_words(df[df['label'] == 0]['content'])
top_real_words = get_most_common_words(df[df['label'] == 1]['content'])


print("\nTop 20 words in FAKE news:")
print(top_fake_words)

print("\nTop 20 words in REAL news:")
print(top_real_words)

print("Step 4 done: EDA complete.")

#Clean the text (NLP preprocessing)


def clean_text(text):
    
    text = str(text).lower()                                  # make everything lowercase
    text = re.sub(r'http\S+', ' ', text)                       # remove links
    text = re.sub(r'\d+', ' ', text)                           # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation

    words = word_tokenize(text)   # break the sentence into a list of words

    clean_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            clean_word = lemmatizer.lemmatize(word)   # turn word into its base form
            clean_words.append(clean_word)

    return ' '.join(clean_words)



test_sentence = "The Government announced 25 NEW policies in 2024!!!"
print("Before cleaning:", test_sentence)
print("After cleaning :", clean_text(test_sentence))


print("\nCleaning all articles... this will take a few minutes, please wait.")
df['clean_text'] = df['content'].apply(clean_text)
print("Step 5 done: text cleaning complete.")

# Convert text into numbers using TF-IDF


X = df['clean_text']
y = df['label']

# split data: 80% ,20%
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Training samples:", len(X_train_text))
print("Testing samples :", len(X_test_text))

# TF-IDF 
tfidf = TfidfVectorizer(max_features=3000)

print("\nFitting TF-IDF... please wait, this can take a minute or two.")
X_train_vectors = tfidf.fit_transform(X_train_text)
X_test_vectors = tfidf.transform(X_test_text)

print("Step 6 done: TF-IDF features created.")
print("Number of features (words) used:", len(tfidf.vocabulary_))


# Train three different models


print("\nTraining Logistic Regression...")
log_reg_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg_model.fit(X_train_vectors, y_train)
print("Logistic Regression trained.")

print("\nTraining Multinomial Naive Bayes...")
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_vectors, y_train)
print("Naive Bayes trained.")

print("\nTraining Linear SVM (this one takes a bit longer)...")
svm_model = LinearSVC(random_state=RANDOM_STATE, max_iter=5000)
svm_model.fit(X_train_vectors, y_train)
print("Linear SVM trained.")

print("Step 7 done: all 3 models are trained.")




def evaluate_model(model, model_name):
    predictions = model.predict(X_test_vectors)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n" + "=" * 50)
    print("Model:", model_name)
    print("=" * 50)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("\nDetailed report:")
    print(classification_report(y_test, predictions, target_names=['FAKE', 'REAL']))

    # confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title(model_name + ' - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return accuracy, precision, recall, f1


acc_lr, prec_lr, rec_lr, f1_lr = evaluate_model(log_reg_model, 'Logistic Regression')
acc_nb, prec_nb, rec_nb, f1_nb = evaluate_model(naive_bayes_model, 'Multinomial Naive Bayes')
acc_svm, prec_svm, rec_svm, f1_svm = evaluate_model(svm_model, 'Linear SVM')

#results
results_table = pd.DataFrame({
    'Model': ['Logistic Regression', 'Multinomial Naive Bayes', 'Linear SVM'],
    'Accuracy': [acc_lr, acc_nb, acc_svm],
    'Precision': [prec_lr, prec_nb, prec_svm],
    'Recall': [rec_lr, rec_nb, rec_svm],
    'F1-Score': [f1_lr, f1_nb, f1_svm],
})

print("\nComparison Table:")
print(results_table)

# simple bar chart comparing accuracy of all 3 models
plt.figure(figsize=(7, 5))
sns.barplot(data=results_table, x='Model', y='Accuracy')
plt.title('Accuracy Comparison of the 3 Models')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

print("Step 8 done: models evaluated and compared.")


# the best model 

model_accuracies = {
    'Logistic Regression': acc_lr,
    'Multinomial Naive Bayes': acc_nb,
    'Linear SVM': acc_svm,
}

all_models = {
    'Logistic Regression': log_reg_model,
    'Multinomial Naive Bayes': naive_bayes_model,
    'Linear SVM': svm_model,
}

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = all_models[best_model_name]

print("Best model is:", best_model_name)
print("Its accuracy is:", round(model_accuracies[best_model_name], 4))


if os.path.isdir('/kaggle/working'):
    output_folder = '/kaggle/working'
else:
    output_folder = '.'

joblib.dump(best_model, os.path.join(output_folder, 'best_fake_news_model.pkl'))
joblib.dump(tfidf, os.path.join(output_folder, 'tfidf_vectorizer.pkl'))

print("Step 9 done: best model and TF-IDF vectorizer saved to disk.")




def predict_news(news_text):
    
    cleaned = clean_text(news_text)
    vector = tfidf.transform([cleaned])

    prediction = best_model.predict(vector)[0]

    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100
    else:
        score = best_model.decision_function(vector)[0]
        confidence = (1 / (1 + math.exp(-abs(score)))) * 100

    label = 'REAL' if prediction == 1 else 'FAKE'

    print("News text  :", news_text[:100])
    print("Prediction :", label)
    print("Confidence :", round(confidence, 2), "%")
    print("-" * 60)

    return label, confidence


print("\nTesting the model on some example headlines:\n")

predict_news("Scientists at NASA confirm successful landing of new rover on Mars.")
predict_news("BREAKING: Aliens spotted controlling world governments through secret towers.")
predict_news("The Reserve Bank of India raised its key interest rate on Wednesday.")
predict_news("Miracle fruit found in Amazon rainforest cures all diseases overnight!")

try:
    user_headline = input("\nEnter a news headline to check (or press Enter to skip): ")
    if user_headline.strip() != '':
        predict_news(user_headline)
    else:
        print("No headline entered, skipping.")
except Exception:
    print("Interactive input is not available here. "
          "You can call predict_news('your headline') directly instead.")

print("\nStep 10 done: prediction function is ready to use.")
print("\nAll steps finished successfully!")
