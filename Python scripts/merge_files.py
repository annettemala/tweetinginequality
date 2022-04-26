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
            'followers count',
            'retweet count',
            'reply count',
            'like count',
            'quote count',
            'frame'
            ]
    df = df[cols] 
    return df

############################
# Merge files
############################
# Define paths
directory_load = working_directory + '/data/analyzed/' 
make_directory_save = working_directory + '/data/fullsample/' 

# Create folder to store data
try:
    os.mkdir(make_directory_save)
except OSError:
    print("Creation of directory failed. It might exist already.")
else:
    print("Successfully created directory.")

# Get single files, analyze and merge them into one
def merge ():
    index = 0
    
    for file in os.listdir(directory_load):
        print("Now processing file ", (index+1))
        # Gets file from directory
        path = os.path.join(directory_load, file)
        df = pd.read_pickle(path)
             
        if index == 0:
            all_tweets = df
            print('Created overall dataset')
        
        else:
            if df.empty:
                # Go to the next file
                print('Dataset empty.')
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
all_tweets.to_pickle(make_directory_save + today + '_tweets_oneyear_analyzed')
print('Saved complete dataframe')

# Save dataframe to Excel
all_tweets.to_excel((make_directory_save + today + '_tweets_oneyear_analyzed') + '.xlsx') 
print('Exported complete dataframe to Excel')
