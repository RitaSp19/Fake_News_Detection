#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score

PATH_TO_TEXT = 'train.tsv'
# выбрать путь к файлу


df = pd.read_csv(PATH_TO_TEXT, sep='\t')
df.head()


labels = df.is_fake
labels.value_counts()


x_train,x_test,y_train,y_test = train_test_split(df.title, labels, test_size = 0.33, random_state = 7)


tfidf_vectorizer = TfidfVectorizer()


tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)


pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)


score = f1_score(y_test, y_pred)
print(f'F1: {round(score * 100,2)}%')
score


PATH_TO_TEST = 'train.tsv'
# выбрать путь к файлу


result = pd.read_csv(RATH_TO_TEST, sep='\t')
result.head()


tfidf_test = tfidf_vectorizer.transform(result.title)


result_pred = pac.predict(tfidf_test)
result_pred


df_result = pd.DataFrame({'title': result.title, 'is_fake': result_pred}) #переименовать
df_result.head()


df_result.to_csv('predictions.tsv', sep="\t")

