# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:19:59 2021

@author: Annette Malapally

"""
import prepare_tweets
import syntax_matcher

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import cohen_kappa_score
from pickle import dump
import statistics 
import matplotlib.pyplot as plt
import eli5
import warnings
from wordcloud import WordCloud

###### Preprocessing ######
# Preprocessing of tweets
def preprocess(df, lemmatize, stem, target_column = 'manual frame'):
    print('Preparing data.')
    
    # Drop duplicate tweets
    df = prepare_tweets.drop_duplicates (df, 'text')
        
    # Lower case
    df['processed text'] = df['text'].str.lower()
    print('     Made tweets lower case.')
    
    # Remove urls
    df = prepare_tweets.remove_urls(df, 'processed text')
    print('     Removed URLs.')
    
    # Remove mentions
    df = prepare_tweets.remove_mentions(df, 'processed text')
    print('     Removed mentions.')
        
    # Remove stopwords
    df = prepare_tweets.remove_stopwords(df, 'processed text', keepthan = True)
    print('     Removed stopwords.')
        
    # Remove punctuation
    df = prepare_tweets.remove_punctuation(df, 'processed text')
    print('     Removed punctuation.')
        
    if lemmatize == True:
        # Lemmatize tweets
        df = prepare_tweets.lemmatize_tweets(df, 'processed text')
        print('     Lemmatized tweets.')
            
    if stem == True:
        # Stem tweets
        df = prepare_tweets.stem_tweets(df, 'processed text')
        print('     Stemmed tweets.')
    
    # Tokenize tweets
    df = prepare_tweets.tokenize_tweets(df, text_column = 'processed text', target_column = 'tokens')
    print('     Tokenized tweets.')
    
    # Random order of tweets
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

###### Vectorizers ######
# Get vectorizer
def get_vectorizer(vec, n, optimized):   
    if vec == 'bow':
        if optimized == True:
            return CountVectorizer(max_df = 0.75)
        else:
            return CountVectorizer()
        
    elif vec == 'ngrams':
        if optimized == True:
            return CountVectorizer(max_df = 0.75, ngram_range=(1, 2))
        else:
            return CountVectorizer(ngram_range=(1,n))
        
    elif vec == 'tfidf':
        if optimized == True:
            return TfidfVectorizer(max_df = 0.75, ngram_range=(1,2))
        else:
            return TfidfVectorizer(ngram_range=(1, n))
        
    else:
        return []

###### Classifiers ######    
# SVC
def lin_svc(optimized):
    classifier = []
    if optimized == False:
        classifier = LinearSVC()
    if optimized == True:
        classifier = LinearSVC(max_iter = 100, penalty  = 'l2', C = 0.5)
    print('Created SVC classifier.')
    return classifier

# Naive Bayes
def complement_bayes():
    classifier = ComplementNB()
    print('Created Complement Naive Bayes classifier.')
    return classifier

# Linear SVM with SGD training
def sgd_svm(optimized):
    if optimized == False:
        classifier = SGDClassifier(loss = "hinge", penalty = "l2")
    if optimized == True:
        classifier = SGDClassifier(loss = "hinge", alpha = 0.00001, max_iter = 1000, penalty = "l2")
    return classifier

# Baseline: Majority classifier
def baseline(y):
    y.value_counts()
    majority = DummyClassifier(strategy='most_frequent')
    print('Created dummy classifier.')
    return majority

# LDA
def lda():
    classifier = LinearDiscriminantAnalysis()
    print('Created LDA classifier.')
    return classifier

# Get classifier
def get_classifier(classify, optimized, y):
    if classify == 'svc':
        return lin_svc(optimized)
    
    elif classify == 'majority':
        return baseline(y)
        
    elif classify == 'cnb':
        return complement_bayes()
    
    elif classify == 'lda':
        return lda()
    
    elif classify == 'sgd':
        return sgd_svm(optimized)
        
    else:
        return 0

###### Train model  ######
# Simple split in training and test data
def split_train_test(X_values, y):
    return train_test_split(
        X_values, y, test_size=0.2, stratify=y
    )

# Find match between frame classified by syntax and by ML
def find_match(df):
    # Find matches
    df['match'] = np.where(df['frame'] == df['ML frame'],1,0)
    return df

###### Compute accuracy  ######  
# Choose method
def compute_accuracy(df, method):
    if method == 'syntax':
        compute_accuracy_syntax(df)
    elif method == 'ml':
        compute_accuracy_ml(df)
    else:
        print('Method unknown.')

# Compute accuracy for syntax method        
def compute_accuracy_syntax(df):
    y_test = df['manual frame']
    y_pred = df['frame']
    
    f = open('accuracy_output_syntax.txt', 'w')
    print(metrics.classification_report(y_test, y_pred), file = f)
    print('Created classification report and printed to file.')
    f.close()
    
    return 0

# Show most informative features
def show_weights(classifier, vectorizer, y_test):
    # run block of code and catch warnings
    with warnings.catch_warnings():
    	# ignore all caught warnings
        warnings.filterwarnings("ignore")
        html_result = eli5.show_weights(classifier, vec = vectorizer, top=100)
    with open('class_weights.html','wb') as f:
        f.write(html_result.data.encode("UTF-8"))

# Output confusion matrix and classification report for ML
def output_accuracy_ml(y_pred, X_test, y_test, classifier, vectorizer, pipe, f):
    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, xticks_rotation=-45)
    plt.savefig('confusionmatrix.png')
    plt.show()
    print('Created confusion matrix and saved to png.')
    
    print(metrics.classification_report(y_test, y_pred), file = f)
    print('Created classification report and printed to file.')
    
    # Most informative features
    show_weights(classifier, vectorizer, y_test)
    
    return 0

# Compute accuracy for machine learning algorithm
def compute_accuracy_ml(df, target, vec, optimized, classify, pipe_name, match, n = 2):  
    print('Running classification.')
    # Dependent variable
    y_true = df[target]
    texts = df['processed text']
        
    # Get vectorizer
    vectorizer = get_vectorizer(vec, n, optimized)
    print('     Got vectorizer: ', vec)
    
    if vectorizer == []:
        print('     No valid embedding given.')
        return 0
       
    # Get classifier
    classifier = get_classifier(classify, optimized, y_true)
    
    if classifier == 0:
        print('     No valid classifier given.')
        return 0

    # Get training and test data        
    X_train, X_test, y_train, y_test = split_train_test(texts, y_true)
    print('     Got training and test data.')
            
    # Make pipeline
    pipe = make_pipeline(vectorizer, classifier)
    
    # Fit data
    pipe.fit(X_train, y_train)
    print('     Trained classifier')
        
    # Predict values to determine accuracy
    y_pred = pipe.predict(X_test) 
    
    # Save predicted values to df
    df['ML frame'] = pipe.predict(texts)
    df = find_match(df)
    print('     Predicted values.')
    
    # Subset of tweets where classification of tweets matches
    if match == True:
        df[df['match'] == 1]
        print('Excluded tweets where syntax and ML classification do not match. Tweets in final df: ', str(len(df)))
    
    # Compute accuracy and print to output file
    f = open('accuracy_output_ml_' + pipe_name + '.txt', 'w')

    print('Classiciation with ',
          'vectorizer: ', vec,
          ', optimized: ', str(optimized),
          ', classifier: ', classify,
          ', data: ', pipe_name,
          ', match of syntax and ML: ', match,
          file = f)
    if (vec == 'ngrams'):
        print('Using ', str(n), '-grams', file = f)
        
    output_accuracy_ml(y_pred, X_test, y_test, classifier, vectorizer, pipe, f)
    
    # Save pipe
    file = open('pipe_' + pipe_name + '.pkl', 'wb')
    dump(pipe, file)
    file.close()
       
    return df

###### Cross-validation across different vectorizers and classifiers ###### 
# Fit data with pipeline
def fit_data(X_train, y_train, X_test, vectorizer, classifier):
    # Make pipeline
    pipe = make_pipeline(vectorizer, classifier)

    # Fit data
    pipe.fit(X_train, y_train)
        
    # Predict values to determine accuracy
    y_pred = pipe.predict(X_test)
    
    return (X_train, y_pred)

# Run cross-validation
def cross_val5(df, group, target, optimized, n = 2):
    # Output file
    f = open('cross_validation_output_' + group + '.txt', 'w')
    
    # True labels
    y_true = df[target]
    
    # Texts
    texts = df['processed text']
    
    # 5-times cross validation validaiton (with equal distribution)
    # Vectorizers
    names_vectorizers = ['bow', 'ngrams', 'tfidf']
    
    # Classifiers
    names_classifiers = ['svc', 'majority', 'cnb']
        
    # Run all combinations
    X_values = []
    for i in range(len(names_vectorizers)):
        vec = names_vectorizers[i]
        print(vec, file = f)
        
        # Get vectorizers
        vectorizer = get_vectorizer(vec, n, optimized)
        X_values = vectorizer.fit_transform(texts)

        # Run classifiers
        for j in range(len(names_classifiers)):
            classify = names_classifiers[j]
            print(classify, file = f)
            classifier = get_classifier(classify, optimized, y_true)
            print("     Overall accuracy", file = f)
            scores = cross_val_score(classifier, X_values, y_true, cv = 5)
            print("          Mean: %0.2f; SD: %0.2f" % (scores.mean(), scores.std()), file = f)
            
            # Get precision and recall scores
            # Iterations
            frames = ['disadvantage frame', 'privilege frame', 'no relevant comparison']
            scoring = ['precision', 'recall']
            
            # Result container
            scores = [[[] for _ in range(len(frames))] for _ in range(len(scoring))]
            
            # Split data into five groups
            skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
            
            # For each group, get training and test data
            for train_index, test_index in skf.split(texts, y_true):
                X_train, X_test = texts[train_index], texts[test_index]
                y_train, y_test = y_true[train_index], y_true[test_index]
                
                # Fit data
                X_train, y_pred = fit_data(X_train, y_train, X_test, vectorizer, classifier)
                
                # Iterate scorings 
                for l in range(len(scores)): 
                    # Iterate frames
                    for m in range(len(frames)):
                        # Get classification report
                        report = metrics.classification_report(y_test, y_pred, output_dict = True)
                        
                        # Get score for current frame
                        score = report[frames[m]][scoring[l]]
                        
                        # Save score in results
                        scores[l][m].append(score)
            
            # Print results to file   
            # Iterate scorings
            for n in range(len(scores)):
                print('     ', scoring[n], file = f)
                # Iterate frames
                for o in range(len(frames)):
                    print('          ', frames[o], file = f)
                    values = scores[n][o]
                    print("               Mean: %0.2f; SD: %0.2f" % (statistics.mean(values), statistics.stdev(values)), file = f)  

    f.close()
    
##### Analyzing topics #######
def remove_keywords(df, file_name):
    kw_adv = 'white'
    kw_dis = 'black'
    other = ('than', 'more', 'less', 'likely', 'asian', 'american', 'latino', 'latina', 'latinx', 'brown', 'african', 'hispanic',
             'people', 'ppl', 'person', 'guy', 'dude', 'women', 'men', 'woman', 'man')
    
    kw_adv = syntax_matcher.define_keywords(kw_adv)
    kw_dis = syntax_matcher.define_keywords(kw_dis)
    
    for kw in (kw_adv, kw_dis):
        for word in other:
            kw.append(word)
    
    df['tokens'] =  [[word for word in tweet if word not in kw_adv] for tweet in df['tokens']]
    df['tokens'] =  [[word for word in tweet if word not in kw_dis] for tweet in df['tokens']]
    
    return df 

def words_list(index, lda_model, vocab, f):   
    #imp_words_topic=""
    comp=lda_model.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:15]
    print(sorted_words, file = f)
    
def word_cloud_sklearn(index, lda_model, vocab, file_group, data_dir):
    imp_words_topic=""
    comp=lda_model.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:15]
    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]
        
    wordcloud = WordCloud(background_color = 'white', width=1800, height=1200).generate(imp_words_topic)
    plt.figure(figsize=(15,15))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(data_dir + '/wordcloud_' + file_group + str(index+1) + '.png')
    print('Created wordcolud ', str(index))

def analyze_topics(df, data_dir, file_group, no_topics = 10):
    df = df.reset_index()
    
    no_features = 1000
    df_topics = pd.DataFrame()
    
    # Prepare tokens
    # Select tweets with relevant frames
    df = df[df['ML frame'] != 'no relevant comparison']
    df = df.reset_index()
    
    # Remove keywords and non-specific words
    df = remove_keywords(df, file_group)
    tokens = df['tokens'].apply(lambda row: " ".join([token for token in row]))
    
    # Feature extraction
    tf_vectorizer = CountVectorizer(max_df=0.75, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(tokens)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
        
    # Topic modeling       
    # Run LDA
    # doc_topic_prior = alpha: a high alpha-value means that each document is likely to contain a 
    # mixture of most of the topics, and not any single topic specifically
    # topic_word_prior = beta:  high beta-value means that each topic is likely to contain a 
    # mixture of most of the words, and not any word specifically
    lda = LatentDirichletAllocation(n_components=no_topics, 
                                    max_iter=5, 
                                    learning_method='online', 
                                    learning_offset=50,
                                    random_state=0,
                                    learning_decay = 0.9,
                                    topic_word_prior = 0.1,
                                    doc_topic_prior = 0.1).fit(tf)
        
    # Get topic distribution per tweet
    topic_distribution = pd.DataFrame(lda.transform(tf))
    
    # Rename columns
    indices = list(range(len(topic_distribution.columns)))
    indices = ['topic_'+ file_group + str(index+1) for index in indices]
    topic_distribution.columns = indices
    
    # Merge data
    df_topics = pd.concat([df, topic_distribution], axis = 1) 
    
    # Diagnose model performance
    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda.score(tf), '(Higher the better)')
    
    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda.perplexity(tf), '(Lower the better)')
    
    # Word cloud
    for index in range(len(topic_distribution.columns)):
        word_cloud_sklearn(index, lda, tf_feature_names, file_group, data_dir)  
        
    # Word list with weights
    # Output file
    f = open('topics_output_race.txt', 'w')
    
    for index in range(len(topic_distribution.columns)):
        words_list(index, lda, tf_feature_names, f)
        
    f.close()
        
    # Get most prominent topic for each tweet
    df_topics['main topic'] = 'NaN'
    df_topics['main topic'] = df_topics[indices].idxmax(axis = 1)
        
    # Grid search for LDA
    #grid_search_lda(tf)
        
    return df_topics

##### Kappa score #######
def get_kappa(df):
    df = df[df['ML frame'] == df['frame']]
    kappa = cohen_kappa_score(df['ML frame'], df['manual frame'])
    return kappa