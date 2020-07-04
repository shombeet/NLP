# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:37:53 2020

@author: sdasgupt
"""
import numpy as np
import pandas as pd
df = pd.read_csv('Spam.csv', encoding='latin=1')

df.head()

df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1, inplace=True)

df.head()

df['Label'] = df['class'].map({'ham':0,'spam':1})

df.head()

X = df['message']
y=df['Label']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

import pickle
pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB
naive = MultinomialNB()

naive.fit(X_train, y_train)
naive.score(X_test,y_test)

filename = 'nlp_model.pkl'
pickle.dump(naive,open(filename,'wb'))