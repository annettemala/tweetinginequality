# tweetinginequality
Analyzes framing of inequality in tweets

To run analysis
1) Get access to the Twitter API on the twitter developer portal and save keys.
	I. Save your access keys in a file named "twitter_keys.txt" in the same folder where the scripts lie. 
	II. The keys must be stored in the following format: key, secret, bearer token in three subsequent, separate lines

2) Make sure that you have installed the following packages on your machine.
- gender_guesser.detector
- nltk
- requests
- sklearn
- spacy and its "en_core_web_sm" model
- textblob
- tweepy
- wordcloud

3) Download all files and folders in this repository and keep the data structure as is; i.e, put all scripts, text files, pickled files and the folder "data" in the same folder on your computer.

4) Use scripts in the following order to go through the analysis steps. please note required user input.
- collect_tweets_archive
	- This script requires user input (variable "group_file"): The topic of inequality for which tweets shall be collected has to be specified: either "money" (for economic inequality) or "race" (for racial inequality).
	- The script then collects tweets from 365 days starting at 01.01.2021.
	- Up to 500 tweets will be collected for each hour in this time span.
	- The collected tweets will be saved as pandas pickle files in the folder '/data/analyzed/' + the group name specified by the user. 
	
- tweet_analysis
	- This script requires user input (variable "group_file"): The topic of inequality for which tweets will be collected has to be specified: either "money" (for economic inequality) or "race" (for racial inequality).
	- The script then prepares tweets for analysis, analyzes the syntax to find inequality frames, predicts the gender of the twitter users, and computes the sentiment of the tweets.
	- The tweets are then saved in the folder '/data/analyzed/' + the group name specified by the user. 

- merge_files
	- This script requires user input (variable "group_file"): The topic of inequality for which tweets will be collected has to be specified: either "money" (for economic inequality) or "race" (for racial inequality).
	- The script then takes all analyzed tweet files and merges them into one dataset.
	- Duplicate tweets are deleted. 
	- The dataframe is then saved in the folder '/data/fullsample' + the group name specified by the user.
	- The file name is: %date%_tweets_oneyear_analyzed_
	- The full dataframe is saved in both pandas pickle and in Excel format. 
	- These files contain a pre-filtered dataset containing potentially relevant tweets and their metrics.

- predict_frames
	- This script predicts frames from a pretrained model. This serves as a final analysis of the tweets' frames.
	- This script requires user input (variable "group_file"): The topic of inequality for which tweets will be collected has to be specified: either "money" (for economic inequality) or "race" (for racial inequality).
	- The topics of the tweets will be analyzed and the topic distribution is saved in the dataset.
	- For each topic, a word cloud is created and saved in '/data/wordclouds/' + the group name specified by the user.
	- The final dataset is saved in Excel format in '/data/fullsample', under the name: '%date%_tweets_oneyear_ML' + the group name specified by the user.

- validate_ML
	- This script requires user input (variable "group_file"): The topic of inequality for which tweets will be collected has to be specified: either "money" (for economic inequality) or "race" (for racial inequality).
	- This script compares the categorization of tweets obtained by the syntax method (tweet_analysis) and the machine learning method (predict_frames) to a set of manually annotated tweets ("tweets_validation.xlsx"). 
	- This script preprocesses tweets by deleting duplicates, converting all text to lower case, removing URLs, mentions, stopwords (except the word "than"), and punctuation. 
	- Then, tweets are lemmatized, tokenized and shuffled.
	- Accuracy (precision and recall) are computed for the syntax and the machine learning method (TF-IDF, SVC) and output in two .txt files: "accuracy_output_" and "accuracy_output_syntax_" + the group name specified by the user. 
	- For the machine learning method, a confusion matrix is output as "confusionmatrix.png", and a table showing ngrams and their weights for the different categories is output as "class_weights.html".
	- A function is provided to compare the accuracy of classification across combinations of different vectorizers and classifiers. Results are output in a .txt file "cross_validation_output_" + the group name specified by the user. 
