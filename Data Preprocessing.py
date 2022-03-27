# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:48:51 2020

@author: rilmia
"""
import pandas as pd
data = pd.read_excel('halodoc_des.xlsx')
data['content'] = data['content'].replace('cepat respon','respon cepat')
data['content'] = data['content'].replace('ramah dokter', 'dokter ramah')
data['content'] = data['content'].replace('cepat dokter', 'dokter cepat')

import unicodedata
import nltk
import re
import string
from nltk.corpus import stopwords
string.punctuation

ADDITIONAL_STOPWORDS = ['ya', 'nya','banget', 'terima','kasih','mantap','top','markotop']
def basic_clean(text):
    words = re.sub(r'[^\w\s]', '', text).split()
    words = [x.lower() for x in words]
    words = {" ".join(sorted(key.split(" "))):text[key] for key in s}
    stopwords = nltk.corpus.stopwords.words('indonesian') + ADDITIONAL_STOPWORDS
    wnl = nltk.stem.WordNetLemmatizer()
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

words = basic_clean(''.join(str(data['content'].tolist())))
(pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]

bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]
bigrams_series
bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

def remDeps(s):
    return {" ".join(sorted(key.split(" "))):s[key] for key in s}
remDeps(basic_clean)

def remove_punctuation(text):
    text_nonpunct = "".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

data['review_clean']  = data['content'].apply(lambda x: remove_punctuation(x))

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

data['review_tokenized'] = data['review_clean'].apply(lambda x: tokenize(x.lower()))

stopword = nltk.corpus.stopwords.words('indonesian')
print(stopword)
def remove_stopwords(tokenized):
    text = [word for word in tokenized if word not in stopword]
    return text

data['review_stopword'] = data['review_tokenized'].apply(lambda x: remove_stopwords(x))

wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized):
    text = [wn.lemmatize(word) for word in tokenized]
    return text

data['review_lemmatized'] = data['review_stopword'].apply(lambda x: lemmatizing(x))

words = data['review_lemmatized']
(pd.Series(nltk.ngrams(words, 2)).value_counts())[:10]

bigrams_series = pd.Series(nltk.ngrams(words, 2)).value_counts()
bigrams_series
bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))