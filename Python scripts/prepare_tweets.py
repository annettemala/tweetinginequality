# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:05:40 2021

@author: Annette Malapally
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import spacy

nlp = spacy.load("en_core_web_sm")

# Removes non-English tweets
def only_english (df):
    # Remove non-English tweets
    df = df[df['language'] == 'en']
    
    # Reset index
    df = df.reset_index(drop=True)

    return df

# Recodes special characters
def recode_chars (df):
    # Recode special characters
    df['text']  = df['text'] .str.replace('&amp;','&', regex = True)
    
    return df

# Finds mentions of "than"
def contains_than(text):
    if (re.search(r"\sthan\s", text)):
        return 'than'
    elif (re.search(r"\sThan\s", text)):
        return 'than'
    if (re.search(r"\sTHAN\s", text)):
        return 'than'
    else: 
        return 'NaN'
   
# Makes sure that each tweet contains the word "than"
def tweets_with_than(df):
    df['than'] = df['text'].apply(contains_than)
    df = df[df['than'] == "than"]
    df = df.drop(['than'], axis = 1)
    
    # Reset index
    df = df.reset_index(drop=True)
    return df

# Removes urls from tweets
def remove_urls(df, column):
    # Remove URLs
    df[column] = df[column].str.replace('http\S+|www.\S+', '', case=False, regex = True)
    
    return df

# Removes mentions from tweets
def remove_mentions(df, column):
    text = df[column]
    
    # Check if given column is text or tokens
    # If it is tokens
    if isinstance(text[0], list):
        # Iterate over rows
        for i in range(len(text)):
            # Remove mentions
            text_row = [re.sub('@[A-Za-z0-9_]+', '', element) for element in text[i]]
            
            # Remove empty strings
            text_row = [element for element in text_row if element != '']

            # Save row
            df[column].iloc[i] = text_row
    
    # If it is text
    else:
        # Remove  mentions
        df[column] = df[column].str.replace('@[A-Za-z0-9_]+', '', case=False, regex = True)
    
    return df

# Make all words lower case
def lower_case(df, column):
    df[column] = df[column].map(lambda x: list(map(str.lower, x)))
    return df

# Splits tweets into sentences
def split_sentences(df):  
    # Split tweets into sentences
    df['sentences'] = df.apply(lambda row: nltk.tokenize.sent_tokenize(row['text cleaned']), axis=1)
    
    # Make sentences lower case
    df = lower_case(df, 'sentences')
    
    return df

# Drops duplicate tweets
def drop_duplicates (df, column = 'text cleaned'):
    # Remove tweets with duplicate text
    df = df.drop_duplicates(subset = column, keep= 'first', inplace = False)
    print('     Dropped duplicate tweets.')
    
    # Reset index
    df = df.reset_index(drop=True)
   
    return df

# Remove stop words, except 'than'
# Define stopwords
stop = set(stopwords.words("english"))
stop_wothan = [item for item in stop if item != 'than']

# Remove stop words in text
def remove_stopwords(df, column, keepthan):
    if keepthan == True:
        stopwords = stop_wothan
    elif keepthan == False:
        stopwords = stop
    df[column] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    return df

# Remove punctuation
def remove_punctuation(df, column):
    df[column] = df[column].str.replace('[^\w\s]','', regex = True)
    return df

# Tokenize tweets
def tokenize_tweets(df, text_column, target_column):
    # Tokenize tweets
    tokens = df[text_column].apply(lambda x: nlp.tokenizer(x))
    df[target_column] = [[token for token in tweet if token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "VERB"] for tweet in tokens]
    df[target_column] = [[token.text for token in tweet if token.text[0] != ' '] for tweet in tokens]
    
    return df

# Lemmatize tweets
def lemmatize_tweets(df, column):
    df[column] = df[column].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
    return df

def stem_tweets(df, column):
    stemmer = SnowballStemmer(language = 'english')
    df[column] = df[column].apply(lambda row: " ".join([stemmer.stem(w.text) for w in nlp(row)]))
    return df
