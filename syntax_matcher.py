# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:29:06 2021

@author: Annette Malapally
"""
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")

# Get keywords from text files
def define_keywords (group):    
    # Get path
    path = group + '.txt'
    
    # Import keywords from file
    with open(path, "r") as file:
        # Read lines in file, where each line is a keyword
        keywords = file.readlines()
        # Remove new lines from keywords
        keywords = [word.strip ('\n') for word in keywords]

    return keywords

keywords_rich = define_keywords('rich')
keywords_poor = define_keywords('poor')
keywords_white = define_keywords('white')
keywords_black = define_keywords('black')

# Clean keywords
def remove_y (keywords):
    cut_keywords = []
    
    for element in keywords:
        cut_element = []
        element = element.split()
        for word in element:
            if word[len(word)-1] == 'y':
                word = word[:-1]
            cut_element.append(word)
        cut_keywords.append(' '.join(cut_element))
    
    return cut_keywords

# Get adjectives from keywords
def get_adjectives(group):
    # Get keywords for given group
    keywords = globals()['keywords_' + group]
    
    # Remove y-endings
    cut_keywords = remove_y(keywords)

    # Split keywords
    adjectives = []
    composites = []
    for element in cut_keywords:
        length = len(element.split())
        if (length==1):
            adjectives.append(element)
        elif (length > 1):
            composites.append(element)
       
    # Convert keywords to lower and upper case versions
    lowup_adj = []      
    for word in adjectives:
        lowup_adj.append('[' + word[0].upper() + word[0] + ']' + word[1:])
        
    lowup_comp = []
    for word in composites:
        lowup_comp.append('[' + word[0].upper() + word[0] + ']' + word[1:])
     
    # Add conjugations and word boundaries
    conjugations = '(y(s|\'s|s\')?|((i)?er)|((i)?est)|((ie)?s|\'s|s\'))?'
    
    conj_adj = []
    for element in lowup_adj:
        conj_adj.append('(' + '\\b' + element + conjugations + '\\b' + ')')
        
    conj_comp = []
    for element in lowup_comp:
        # Split composite into words
        element = element.split()
        length = len(element)
        
        # Compose 
        # Open parenthesis
        composite = '('
        # Add each word with conjugations, followed by white space
        for i in range(length-1):
            composite = composite + ('(' + '\\b' + element[i] + conjugations + '\\b' + ')' 
                                  + '(\s)')
        # Add last word and close parenthesis
        composite = composite + ('(' + '\\b' + element[length-1] + conjugations + '\\b' + ')' + ')')
        
        conj_comp.append(composite)
        
    # Compose regex pattern
    adjectives = '(' 
    
    if conj_adj != []:
        adjectives = adjectives + '(' + '|'.join(conj_adj) + ')' 

    if ((conj_adj != []) and (conj_comp != [])):
        adjectives = adjectives + '|' 

    if conj_comp != []:
        adjectives = adjectives + '(' + '|'.join(conj_comp) + ')' 

        
    adjectives = adjectives + ')'

    return adjectives

# Get nouns
def get_nouns():
    keywords = define_keywords('nouns')
    
    cut_keywords = remove_y(keywords)
    
    # Add conjugations and word boundaries
    nouns = []
    for element in cut_keywords:
        nouns.append('(' + '\\b' + element + '(y|(ie)?s)?' + '\\b' + ')')
        
    # Compose regex pattern
    nouns = '(' + '|'.join(nouns) + ')'
    
    return nouns 

# Compose regex term    
def keywords_to_regex(group):
    # Get adjectives
    adjectives = get_adjectives(group)
 
    pat_keywords = adjectives
    return pat_keywords

# Recode frames: consider only tweets where only one type of frame was found
def recode_frames(frames):
    # Check if any relevant frames where found
    if frames == []:
        return 'no relevant comparison'
    else:    
        # Convert frames to integers
        frames = [frame for frame in frames if frame != 'no relevant comparison']
        
        # Get only unambigious frames 
        length = len(set(frames))
        if length == 1:
            return frames[0]
        else:
            return 'no relevant comparison'

# Find tweets which match the given pattern
def get_frames(sentence, priv_group, disadv_group):
    doc = nlp(sentence)
    
    # Get regex patterns
    regex_pat_dg = keywords_to_regex(disadv_group)
    regex_pat_pg = keywords_to_regex(priv_group)

    pattern_df = [{'OP': '*'},
                  {'TEXT': {'REGEX': regex_pat_dg}, 
                   'DEP': {'IN': ['nsubj', 'dobj', 'nsubjpass']}
                   },
                  {'OP': '*'},
                  {'ORTH': 'than'},
                  #{'OP': '*'},
                  {'TEXT': {'REGEX': regex_pat_pg},
                   'DEP': 'pobj'}
                  ]

    pattern_pf = [{'OP': '*'},
                  {'TEXT': {'REGEX': regex_pat_pg}, 
                   'DEP': {'IN': ['nsubj', 'dobj', 'nsubjpass']}
                   },
                  {'OP': '*'},
                  {'ORTH': 'than'},
                  #{'OP': '*'},
                  {'TEXT': {'REGEX': regex_pat_dg},
                   'DEP':  'pobj'}
                  ]
    
    # Create matcher
    matcher = Matcher(nlp.vocab)

    matcher.add("disadvantage frame", [pattern_df])
    matcher.add("privilege frame", [pattern_pf])
        
    # Get matching patterns    
    matches = matcher(doc)
    frames = []
    
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        frames.append(string_id)
 
    # Recode frames
    frames = recode_frames(frames)
    return (frames)

# Find frame for each sentence in a tweet
def iterate_sentences(sentences, priv_group, disadv_group):
    frames = []
    # Checks if sentence is empty
    empty = []
    if (sentences == empty):
        print('     No sentences found.')
        return 'no relevant comparison'
        
    # Finds frames in all sentences in a tweet
    [frames.append(get_frames(sentence, priv_group, disadv_group)) for sentence in sentences]

    # Recode frames
    frames = recode_frames(frames)
        
    return (frames)

# Create variable and store found frame for each tweet in it
def create_var_frame(df, priv_group, disadv_group, column):    
    # Create variable
    df[column] = 'no relevant comparison'
    
    # Analyze frame
    for i in range(len(df)):
        # Get sentences
        sentences = df['sentences'].iloc[i]
        
        # Get frame and group
        result = iterate_sentences(sentences, priv_group, disadv_group)
        frame = result
        
        # Save frame
        df[column].iloc[i] = frame
    
    return df

# Find frames for given group
def find_frames(df, group):
    if group == 'race':
        group_list = ['white', 'black']
    
    elif group == 'gender':
        group_list = ['rich', 'poor']
        
    else:
        print('          Groups unknown.')

    # Create variable
    df = create_var_frame(df, group_list[0], group_list[1], 'frame')
    print('          Analyed frames for: ', group_list, ' in all tweets')
        
    return df