import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


dataset = pd.read_csv('sentiment.csv')


def untag(tweet: str):
    return re.sub("@\w+", "", tweet)


def unlink(tweet: str):
    unlinked = re.sub("http?://[^\s]+", "", tweet)
    return re.sub("https?://[^\s]+", "", tweet)


stop_words = set(stopwords.words('english'))
stop_words.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                  'nine', 'ten', 'may', 'also', 'across', 'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)


def remove_stop_words(tweet):
    return re_stop_words.sub("", tweet)


stemmer = SnowballStemmer("english")


def stemming(tweet):
    stem_sentence = ""
    for word in tweet.split(" "):
        stem = stemmer.stem(word)
        stem_sentence += stem
        stem_sentence += " "
    stem_sentence = stem_sentence.strip()
    return stem_sentence


dataset['Tweet'] = dataset['Tweet'].str.lower()
dataset['Tweet'] = dataset['Tweet'].apply(untag)
dataset['Tweet'] = dataset['Tweet'].apply(unlink)
dataset['Tweet'] = dataset['Tweet'].apply(remove_stop_words)
dataset['Tweet'] = dataset['Tweet'].apply(stemming)

y = dataset['Emotion']
X = dataset['Tweet']


vectorizer = TfidfVectorizer(
    strip_accents='unicode', analyzer='word', ngram_range=(1, 3), norm='l2')


train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1, test_size=0.20, shuffle=True)

vectorizer.fit(X)

train_X = vectorizer.transform(train_X)
val_X = vectorizer.transform(val_X)

model = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)


def train():

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    mae = mean_absolute_error(preds, val_y)

    print(1 - mae)


train()
