# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:15:29 2022

@author: Annette Malapally
"""

import pandas as pd
import datetime
import os
from pickle import load

# Define which groups should be analyzed
group_file = 'race'

# Define current date
today = datetime.datetime.today() 
today = today.strftime("%Y-%m-%d")

# Define path where folder with data lie
working_directory = os.path.dirname(os.path.realpath(__file__))
data_directory = (working_directory + '/data/fullsample/' + group_file + '/') 
data_directory_clouds = (working_directory + '/data/wordclouds/' + group_file + '/')

# Choose working directory
os.chdir(working_directory)

import train_ML

################# Get data #################
# Read data
def get_data():
    for file in os.listdir(data_directory):
        #Gets file from directory
        path = os.path.join(data_directory, file)
        df = pd.read_pickle(path)
        return df
        
df_tweets = get_data()

# Subset 
group_file = 'race'

################# Get pre-trained pipe #################
print('Loading pre-trained model.')
file = open('pipe_' + group_file + '.pkl', 'rb')
pipe = load(file)
file.close()

################# Prepare data #################
df_tweets = train_ML.preprocess(df_tweets, lemmatize = True, stem = False, target_column = 'frame')

################# Predict frame #################
print('Predicting frames.')
df_tweets['ML frame'] = pipe.predict(df_tweets['processed text'])
df_tweets = df_tweets[df_tweets['ML frame'] != 'no relevant comparison']

################# Analyze topics #################
print('Analyzing topics of tweets with relevant frames.')
df_topics = train_ML.analyze_topics_lda(df_tweets, data_directory_clouds, file_group = group_file, no_topics = 8)

################# Save data #################
print('Saving data.')
df_topics.to_excel((data_directory + today + '_tweets_oneyear_ML_' + group_file) + '.xlsx') 

print('All done.')