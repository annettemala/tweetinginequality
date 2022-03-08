# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:19:59 2021

@author: Annette Malapally

"""

import pandas as pd
import os

# Define path where folder with data lie
working_directory = os.path.dirname(os.path.realpath(__file__))
data_directory = (working_directory + '/data/validation/') 

# Choose working directory
os.chdir(working_directory)

import train_ML

################# Get data #################
# Read data
file_name = 'race'
df_tweets = pd.read_excel(data_directory + 'tweets_validation_' + file_name + '.xlsx')

################# Run analysis #################
# Process tweets
df_tweets = train_ML.preprocess(df_tweets, lemmatize = True, stem = False)

# Accuracy of syntax classification
train_ML.compute_accuracy_syntax(df_tweets)

# Accurarcy of ML classification
df_tweets = train_ML.compute_accuracy_ml(df_tweets, 
                    target = 'manual frame',
                    vec = 'tfidf',
                    optimized  = True, 
                    classify = 'svc',
                    pipe_name = file_name,
                    match = False, 
                    n = 2
                    )  

# 5-fold cross-validation of tweet categorization
train_ML.cross_val5(df_tweets, 
                    group = file_name,
                    target = 'manual frame',
                    optimized = True,
                    n = 2
                    )