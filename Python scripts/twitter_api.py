# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 08:59:10 2021

@author: Annette Malapally
"""

import tweepy as tw

# Import Twitter keys from given path
def import_keys (path):  
    # Open file
    with open(path, "r") as file:
        # Read lines in file, where each line is a keyword
        keys = file.readlines()
        # Remove new lines from keywords
        keys = [word.strip ('\n') for word in keys]

    return keys

# Get twitter access
def get_api():
    keys = import_keys('twitter_keys.txt')
    
    # Get access keys
    consumer_key= keys[0]
    consumer_secret= keys[1]
    access_token= keys[2]
    access_token_secret= keys[3]
    
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    
    return api

# Get bearer token
def get_bearer_token():
    keys = import_keys('twitter_keys.txt')
    bearer = keys[2]
    
    return bearer
