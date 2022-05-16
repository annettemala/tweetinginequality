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
import syntax_matcher

nlp = spacy.load("en_core_web_sm")

############################
# Prepare df for analysis
############################
def prepare (df):
    print('     Preparing data')
    # Keep only tweets in English
    df = prepare_tweets.only_english(df)
    
    # Recode special characters
    df = prepare_tweets.recode_chars(df)
    
    # Keep only tweets containing the word "than"
    df = prepare_tweets.tweets_with_than(df)
               
    # Remove urls
    df['text cleaned'] = df['text']
    df = prepare_tweets.remove_urls(df, 'text cleaned')
    
    # Remove mentions from tweets
    df = prepare_tweets.remove_mentions(df, 'text cleaned')
            
    # Remove leading whitespaces from tweets
    df['text cleaned'] = df['text cleaned'].str.strip()
    
    # Split tweets into sentences, in lower case  
    df = prepare_tweets.split_sentences(df)
       
    return df
    
############################
# Analyze focus
############################
def analyze_focus(df):
    print('     Analyzing focus')
    
    # Find comparison focus
    df = syntax_matcher.find_frames(df)

    # Only include cases where frame was found
    df = df[df['frame']!='no relevant comparison']
    print('          Deleted tweets where no frame was found')
        
    return df

############
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
# Analyze files
############################
# Define paths
data_directory = working_directory + '/data/raw/' 
make_directory_save = working_directory + '/data/analyzed/' 

# Create folder to store data
try:
    os.mkdir(make_directory_save)
except OSError:
    print("Creation of directory failed. It might exist already.")
else:
    print("Successfully created directory.")

# Get single files, analyze them
def analyze_tweets ():
    index = 0
    
    # Iterate over files
    day = 0
    hour = 0

    for file in os.listdir(data_directory):
        print("Now processing file ", (index+1))
        #Gets file from directory
        path = os.path.join(data_directory, file)
        df = pd.read_pickle(path)
             
        if df.empty:
            # Go to the next file
            index += 1
            
        else: 
            # Prepare data
            df = prepare(df)
            
            # Analyze focus
            df = analyze_focus(df)
            
            # Drop duplicate tweets, keeping first tweet with identical text
            df = prepare_tweets.drop_duplicates(df)
            
        # Save intermediate result
        df.to_pickle(make_directory_save + today + '_tweets_' + str(day) + '_' + str(hour))
             
        # Go to the next file
        index += 1
        
        # Update day and hour
        hour +=1
        if hour == 24:
            hour = 0
            day += 1
            
    return 0

analyze_tweets()