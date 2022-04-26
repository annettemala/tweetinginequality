# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 08:25:11 2021

@author: Annette Malapally
"""

import requests
import os
import datetime
import time
import pandas as pd

# Define path where folder with data lie
working_directory = os.path.dirname(os.path.realpath(__file__))
make_directory = (working_directory + '/data/raw/') 

# Choose working directory
os.chdir(working_directory)

import twitter_api

# Create folder to store data
try:
    os.mkdir(make_directory)
except OSError:
    print("Creation of directory failed. It might exist already.")
else:
    print("Successfully created directory.")

# Get date
today = datetime.datetime.today().strftime("%Y-%m-%d")

# Get twitter bearer token
bearer = twitter_api.get_bearer_token()

# Define search twitter function
def search_twitter(query, start_time, end_time, tweet_fields, user_fields, expansions, max_results, bearer_token):
    headers = {'Authorization': 'Bearer {}'.format(bearer_token)}
    url = "https://api.twitter.com/2/tweets/search/all?query={}&{}&{}&{}&{}&{}&{}".format(
        query, start_time, end_time, tweet_fields, user_fields, expansions, max_results
    )
    response = requests.request("GET", url, headers=headers)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
        
    return response.json()

# Get tweets
# Get date
def get_dates(counter, hours):  
    # Get start date
    first_day = datetime.datetime(2021, 1, 1, 0, 0, 0, 0)
    date_start = ((first_day + datetime.timedelta(counter)).replace(hour = hours, minute = 0, second = 0, microsecond = 0).isoformat()) + 'Z'
    date_end = ((first_day + datetime.timedelta(counter)).replace(hour = hours, minute = 59, second= 59, microsecond = 0).isoformat()) + 'Z'
    
    start_time = 'start_time=' + date_start
    end_time = 'end_time=' + date_end
    
    return (start_time, end_time)

# Define search terms
def define_query_term(path):
    # Open file
    with open(path, "r") as file:
        # Read lines in file, where each line is a keyword
        keywords = file.readlines()

    return keywords[0]


def collect_tweets(counter, dates):   
    # Get search terms
    query = define_query_term('keywords_race.txt')

    # Define tweet and user info to collect
    tweet_fields = 'tweet.fields=text,author_id,created_at,public_metrics,lang'
    user_fields = 'user.fields=name,username,entities,public_metrics'
    expansions = 'expansions=author_id'
    
    # Define timespan
    start_time = dates[0]
    end_time = dates[1]
    
    # Define number of results
    max_results = 'max_results=500'

    # Call twitter api
    result = search_twitter(query, start_time, end_time, tweet_fields, user_fields, expansions, max_results, bearer) 
    
    # Convert to pandas df
    data = result['data']
    
    try:
        tweets = pd.json_normalize(data)
        users = pd.json_normalize(result['includes']['users'])[['username', 'id', 'name', 'public_metrics.followers_count']]
        tweets_df = tweets.merge(users, left_on = 'author_id', right_on = 'id')
        tweets_df = tweets_df.drop(['id_y'], axis = 1)
    
        # Rename and order df
        tweets_df = tweets_df.rename(columns = {'author_id': 'author id',
                                                'created_at' : 'date',
                                                'text': 'text',
                                                'lang': 'language',
                                                'id_x': 'tweet id',
                                                'public_metrics.retweet_count' : 'retweet count',
                                                'public_metrics.reply_count' : 'reply count',
                                                'public_metrics.like_count' : 'like count',
                                                'public_metrics.quote_count' : 'quote count',
                                                'public_metrics.followers_count' : 'followers count'})
        
        tweets_df = tweets_df[['text', 'date', 'language', 'tweet id', 'author id', 'username', 'name', 'followers count', 'retweet count', 'reply count', 'like count', 'quote count']]
        
        return tweets_df
    
    except KeyError:        
        return None

# Collect tweets within a year
def collect_year_of_tweets():
    request_counter = 0

    # Iterate over the year
    days_counter = 291
    hours = 7
        
    while days_counter < 365:
        # Print status
        print('Collecting tweets from day: ', str(days_counter+1))
               
        # For every hour in the day
        while hours < 24:
            # Get start and end date
            dates = get_dates(days_counter, hours) 
            
            # Get tweets
            tweets = collect_tweets(days_counter, dates)
            
            if len(tweets) > 0:
                print('     Collected ', len(tweets), ' tweets from ', hours, ':00 to ', hours, ':59h.')
                     
                # Save result
                save_path = (make_directory + '/' + today + '_tweets_raw_' + str(days_counter) + '_' + str(hours))
                tweets.to_pickle(save_path)
                
            else:
                print('     Found no tweets.')
            
            # Update hour
            hours += 1
            
            # Update request counter
            request_counter += 1
            
            # Check rate limit
            if request_counter == 300:
                print('Rate limit reached. Waiting for 15 minutes')
                # Reset request counter
                request_counter = 0
                time.sleep(900)
                
            # Wait for rate limit
            time.sleep(1) 
            
        # Update day counter
        days_counter += 1
        
        # Reset hours
        hours = 0
                    
collect_year_of_tweets()       
