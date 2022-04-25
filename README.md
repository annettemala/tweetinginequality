To run analysis
1) Get access to the Twitter API on the twitter developer portal and save keys.
	I. Save your access keys in a file named "twitter_keys.txt" in the same folder where the scripts lie. 
	II. The keys must be stored in the following format: key, secret, bearer token in three subsequent, separate lines

2) Make sure that you have installed the following packages on your machine.
	gender_guesser.detector
	nltk
	requests
	sklearn
	spacy and its "en_core_web_sm" model
	tweepy
	wordcloud

3) Put all scripts, text files, pickled files and the folder "data" in the same folder on your computer.

4) Use scripts in the following order to go through the analysis steps. Please note required user input.
	I. collect_tweets_archive
	   The script collects tweets from 365 days starting at the date specified by the user. Default is January, 1st, 2021. 
	   Up to 500 tweets will be collected for each hour in this time span (less than 500 tweets will be collected if fewer tweets match the query).
	   The collected tweets will be saved as pandas pickle files in the folder '/data/analyzed/'. 
	
	II. tweet_analysis
	   This script prepares tweets for analysis, and analyzes the syntax to find inequality frames.
	   The tweets are then saved in the folder '/data/analyzed/'. 

	III. merge_files
	   This script takes all analyzed tweet files and merges them into one dataset.
	   Duplicate tweets are deleted. 
	   The dataframe is then saved in the folder '/data/fullsample'.
	   The file name is the current date + "_tweets_oneyear_analyzed".
	   The full dataframe is saved in both pandas pickle and in Excel format. 
	   These files contain a pre-filtered dataset containing potentially relevant tweets and their metrics.

	IV. predict_frame_ML
	   This script predicts frames from a pretrained model. This serves as a final categorization of the tweets' frames.
	   The topics of the tweets is analyzed and the topic distribution is saved in the dataset.
	   For each topic, a word cloud is created and saved in '/data/wordclouds/'.
	   The final dataset is saved in Excel format in '/data/fullsample', under the name: current date + "_tweets_oneyear_ML".

	V. validate_ML
	   This script compares the categorization of tweets obtained by the syntax method (tweet_analysis) and the machine learning method (predict_frames) to a set of manually annotated tweets ("tweets_validation.xlsx"). 
	   This script preprocesses tweets by deleting duplicates, converting all text to lower case, removing URLs, mentions, stopwords (except the word "than"), and punctuation. 
	   Then, tweets are lemmatized, tokenized and shuffled.
	   Accuracy (precision and recall) are computed for the syntax and the machine learning method (TF-IDF, SVC) and output in two .txt files: "accuracy_output" and "accuracy_output_syntax". 
	   For the machine learning method, a confusion matrix is output as "confusionmatrix.png", and a table showing ngrams and their weights for the different categories is output as "class_weights.html".
	   A function is provided to compare the accuracy of classification across combinations of different vectorizers and classifiers. Results are output in a .txt file "cross_validation_output". 
