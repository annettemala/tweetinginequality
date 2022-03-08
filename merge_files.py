# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:19:59 2021

@author: Annette Malapally

"""
import os
import datetime

# Define current date
today = datetime.datetime.today() 
today = today.strftime("%Y-%m-%d")

# Define path where folder with data lie
working_directory = os.path.dirname(os.path.realpath(__file__))

# Choose working directory
os.chdir(working_directory)

import pandas as pd
import spacy
import prepare_tweets

nlp = spacy.load("en_core_web_sm")

# Decide which groups to analyze
group_file = 'race'

############################
# Order df
############################
# Order columns of dataframe
def order_df(df):
    cols = ['text', 
            'date',
            'tweet id',
            'author id',
            'username',
            'name',
            'gender',
            'followers count',
            'retweet count',
            'reply count',
            'like count',
            'quote count',
            'frame',
            'polarity', 'subjectivity'
            ]
    df = df[cols] 
    return df

############################
# Merge files
############################
# Define paths
data_directory = working_directory + '/data/analyzed/' + group_file + '/'
data_directory_save = working_directory + '/data/fullsample/' + group_file + '/'

# Get single files, analyze and merge them into one
def merge ():
    index = 0
    
    for file in os.listdir(data_directory):
        print("Now processing file ", (index+1))
        #Gets file from directory
        path = os.path.join(data_directory, file)
        df = pd.read_pickle(path)
        print('     ', path)
             
        if df.empty:
            # Go to the next file
            index += 1
        
        else:
            if index == 0:
                all_tweets = df
                print('Created overall dataset')
            
            else:
                # Add file to dataframe
                all_tweets = pd.concat([all_tweets, df])
                print("     Added file ", (index+1), ' to the dataframe')
                                     
            # Go to the next file
            index += 1
    
    
    # Order columns of df
    all_tweets = order_df(all_tweets)

    return all_tweets

all_tweets = merge()

# Drop duplicate tweets, keeping first tweet with identical text
all_tweets = prepare_tweets.drop_duplicates(all_tweets, column = 'text')

############################
# Save df
############################
# Save dataframe for Python
all_tweets.to_pickle(data_directory_save + today + '_tweets_oneyear_analyzed')
print('Saved complete dataframe')

# Save dataframe to Excel
all_tweets.to_excel((data_directory_save + today + '_tweets_oneyear_analyzed') + '.xlsx') 
print('Exported complete dataframe to Excel')