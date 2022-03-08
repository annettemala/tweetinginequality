# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:51:26 2021

@author: Annette Malapally
"""

import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

# This function calculates the polarity and subjectivity score for each tweet
def sentiment_scores(tweet):
    tweet_sentiment = TextBlob(tweet).sentiment
    polarity = round(tweet_sentiment[0],2)
    subjectivity = round(tweet_sentiment[1],2)
    return (polarity, subjectivity)

# This function adds variables containing the senitment scores to the dataframe
def create_vars_sentiment (df):
    df['sentiment_scores'] = df['text'].apply(sentiment_scores)
    df.loc[:, 'polarity'] = df['sentiment_scores'].map(lambda x: x[0])
    df.loc[:, 'subjectivity'] = df['sentiment_scores'].map(lambda x: x[1])
    df = df.drop(['sentiment_scores'], axis = 1)
    return df