# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:22:45 2020

@author: zhaoh
"""

import os
import sys
import re
import joblib
import numpy as np
import pandas as pd
import nltk
import torch
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def clean_text(var):
    tmp = re.sub('[^A-z-]+', ' ', var).lower()
    return tmp

def extract_senti(var):
    analyser = SentimentIntensityAnalyzer()
    tmp_score = analyser.polarity_scores(var)
    tmp_out = tmp_score['compound']
    return tmp_out

def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines

def count_word_matrix(tweets,word_list):
    my_matrix = pd.DataFrame()
    for text in tweets:
        my_dic = dict()
        text_list = text.split()
        for word in word_list:
            my_dic[word] = text_list.count(word)
        my_matrix = my_matrix.append(my_dic, ignore_index=True)
    my_matrix = np.array(my_matrix)
    return my_matrix

## ----------------------------------------------------------------------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def remove_leading_usernames(tweet):
    """
        Remove all user handles at the beginning of the tweet.

        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """
    regex_str = '^[\s.]*@[A-Za-z0-9_]+\s+'

    original = tweet
    change = re.sub(regex_str, '', original)

    while original != change:
        original = change
        change = re.sub(regex_str, '', original)

    return change

def process_tweet(tweet):
    """
        Preprocess tweet. Remove URLs, leading user handles, retweet indicators, emojis,
        and unnecessary white space, and remove the pound sign from hashtags. Return preprocessed
        tweet in lowercase.

        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """

    #Remove www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))\s+','',tweet)
    tweet = re.sub('\s+((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    #Remove RTs
    tweet = re.sub('^RT @[A-Za-z0-9_]+: ', '', tweet)
    # Incorrect apostraphe
    tweet = re.sub(r"’", "'", tweet)
    #Remove @username
    tweet = remove_leading_usernames(tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace ampersands
    tweet = re.sub(r' &amp; ', ' and ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    #Remove emojis
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet.lower().strip()

def bert_tokenize(tweet):
    """
        Use the global BERT tokenizer to tokenize tweet and return list of tokens.

        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """
    sentences = nltk.tokenize.sent_tokenize(tweet)
    string = "[CLS] "
    for sent in sentences:
        string += process_tweet(sent) + ' [SEP] '
    return tokenizer.tokenize(string)

def bert_token_ids(tweet):
    """
        Tokenize the tweet and return a list of BERT token IDs

        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """

    tokenized_text = bert_tokenize(tweet)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    return indexed_tokens

def bert_segments(tweet):
    """
        Return the sentence segment IDs corresponding to the BERT tokenization
        of the tweet.

        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """
    tokenized_text = bert_tokenize(tweet)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    idx = indexed_tokens.index(102)
    segments_ids = [0]*(idx+1) + [1]*(len(indexed_tokens)-idx-1)

    return segments_ids

def bert_tokens_arr(tweet_series):
    """
        Returns a tuple with (1) a 2-dimensional array of token IDs for the BERT
        tokenization of each tweet in tweet_series; and (2) the mask for ignoring
        the array padding.

        Parameters
        -----------------
        tweet_series : array-like, a list of tweet strings
    """
    tokens = [bert_token_ids(txt) for txt in tweet_series]
    lens = [len(l) for l in tokens]
    maxlen=max(lens)
    tokens_arr = np.zeros((len(tokens),maxlen),int)
    mask = np.arange(maxlen) < np.array(lens)[:,None]
    tokens_arr[mask] = np.concatenate(tokens)
    return tokens_arr, mask.astype('int')

def bert_segments_arr(tweet_series):
    """
        Returns a 2-dimensional array segment IDs for the BERT tokenization of
        each tweet in tweet_series.

        Parameters
        -----------------
        tweet_series : array-like, a list of tweet strings
    """
    segments = [bert_segments(txt) for txt in tweet_series]
    lens = [len(l) for l in segments]
    maxlen=max(lens)
    segments_arr = np.ones((len(segments),maxlen),int)
    mask = np.arange(maxlen) < np.array(lens)[:,None]
    segments_arr[mask] = np.concatenate(segments)
    return segments_arr

def bert_feat_batch(tweet_batch, model, cuda=False):
    """
        Embeds each tweet in tweet_batch into a 768-dimensional vector and returns a
        2-dimensional numpy array of the embeddings from the per-tweet average of the
        BERT word embeddings for each word in the tweet.

        Parameters
        -----------------
        tweet_batch : array-like, a list of tweet strings
        model : a pretrained BertModel, used to generate embeddings
        cuda : boolean, whether to use GPU if it is available
    """
    tok_arr, msk_arr = bert_tokens_arr(tweet_batch)
    seg_arr = bert_segments_arr(tweet_batch)

    # convert to torch tensor with size truncated to fit BERT (seq 512 characters max)
    tok = torch.LongTensor(tok_arr[:, :512])
    msk = torch.LongTensor(msk_arr[:, :512])
    seg = torch.LongTensor(seg_arr[:, :512])

    if (torch.cuda.is_available()) and cuda:
        tok = tok.to('cuda')
        seg = seg.to('cuda')
        msk = msk.to('cuda')

    with torch.no_grad():
        encoded_layers, _ = model(tok, seg, msk)
        return torch.mean(encoded_layers[11], dim=1)

def bert_features(tweet_arr, batch_size=100, cuda=True):
    """
        Batch processes tweet_arr to generates 768-dimensional tweet embeddings. Returns
        2-dimensional numpy array of the embeddings from the per-tweet average of the
        BERT word embeddings for each word in the tweet.

        Parameters
        -----------------
        tweet_arr : array-like, a list of tweet strings
        batch_size : int, number of tweets to use in each batch
        cuda : boolean, whether to use GPU if it is available
    """
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    if (torch.cuda.is_available()) and cuda:
        print("Using GPU")
        model = model.to('cuda')

    outputs = []
    for i in range(0, len(tweet_arr), batch_size):
        outputs.append( np.array(bert_feat_batch(tweet_arr[i: i+batch_size], model, cuda).cpu()) )

    torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0)

def hate_vec(text, regex_list):
    """
        Generates a feature vector for text with one hot encodings to represent whether each
        regular expression in regex_list was a match.

        Parameters
        -----------------
        text : str, any string
        regex_list : array-like, a list of compiled regular expressions
    """
    text = text.lower()
    vec = np.zeros(len(regex_list))
    for i, regex in enumerate(regex_list):
        if len(regex.findall(text)) > 0:
            vec[i] = 1
    return vec

def hate_features(tweet_series):
    """
        Returns a 2-dimensional numpy array of hate_vec feature vectors for each tweet in
        tweet_series.

        Parameters
        -----------------
        tweet_series : array-like, a list of string representations of tweets
    """

    hate_certain = ['aseng', 'bamboo coon', 'bamboo coons', 'bat eater', 'bioterrorism', 'bioweapon', 'boycottchina', 'ccpvirus', 'chinadidthis', 'chinaliedpeopledie', 'chinaliedpeopledied', 'chinaman', 'chinamen', 'chinavirus', 'ching chong', 'chink', 'chinky', 'cokin', 'commie', 'commies', 'communistvirus', 'coolie', 'dog eater', 'fuckchina', 'kungflu', 'ling ling', 'makechinapay', 'niakoué', 'pastel de flango', 'sideways cooters', 'sideways pussies', 'sideways pussy', 'sideways vagina', 'sideways vagina', 'slant-eye', 'slopehead', 'ting tong', 'wuflu', 'wuhanflu', 'wuhanvirus']
    hate_possible = ['asia', 'asian', 'beijing', 'ccp', 'china', 'chinese', 'ckmb', 'communist', 'communists', 'cpc', 'huanan', 'hubei', 'jinping', 'patient zero', 'prc', 'tedros', 'wuhan', 'xi jinping', 'xijinping', 'xinnie']
    check_terms = hate_certain + hate_possible
    regex_list = [re.compile(word) for word in check_terms]

    return np.array([hate_vec(txt, regex_list) for txt in tweet_series])


def label(row):
    """
        Returns a single string label to capture the class of the tweet given in row.
        A 'Hate' tweet is one with greater than a 50% chance of being hateful, less
        than a 50% chance of being counterhate, and less than 50% chance of being neutral, etc.

        Parameters
        -----------------
        row : a row in a Pandas dataframe containing the fields 'Hate Probability',
        'Counterhate Probability', and 'Neutral Probability' (from the covid model)
    """

    if (row['Hate Probability']>=0.5) and (row['Counterhate Probability']<0.5) and (row['Neutral Probability']<0.5):
        return 'Hate'
    elif (row['Hate Probability']<0.5) and (row['Counterhate Probability']>=0.5) and (row['Neutral Probability']<0.5):
        return 'Counterhate'
    elif (row['Hate Probability']<0.5) and (row['Counterhate Probability']<0.5) and (row['Neutral Probability']>=0.5):
        return 'Neutral'
    else:
        return 'Other'
    
## ----------------------------------------------------------------------

if __name__ == '__main__':

    usage = "Usage: python covid_classifier.py [input_file] [output_file]"
    if len(sys.argv) != 3:
        print(usage)
        sys.exit()
        
        
    negative_path = "D:/mystuff/covid-hate-tweet-data-and-classifier/covid-hate/classifier/negative.txt"
    positive_path = "D:/mystuff/covid-hate-tweet-data-and-classifier/covid-hate/classifier/positive.txt"
    
    negative_list = ReadTxtName(negative_path)
    positive_list = ReadTxtName(positive_path)
    
    in_fn = sys.argv[1]
    print(in_fn)
    out_fn =  sys.argv[2]
    print(out_fn)
    path_in = "D:/mystuff/covid-hate-tweet-data-and-classifier/covid-hate/classifier/"
    dirname = os.path.dirname(__file__)
    print("reading csv ...")
    df = pd.read_csv(in_fn,dtype=str)
    tweets = df.full_text
    tweets = tweets.apply(clean_text)
    senti = tweets.apply(extract_senti)
    senti = np.array(senti)
    
    
    negative_matrix = count_word_matrix(tweets,negative_list)
    positive_matrix = count_word_matrix(tweets,positive_list)
    
    indicator = senti >= 0
    
    row_num = 0
    for row in indicator:
        if (row):
            negative_matrix[row_num,:] = -1 * negative_matrix[row_num,:]
        else:
            positive_matrix[row_num,:] = -1 * positive_matrix[row_num,:]
        row_num = row_num + 1
    
    abs_senti = np.mat(abs(senti)).transpose()
    
    negative_matrix = np.multiply(abs_senti,negative_matrix)
    positive_matrix = np.multiply(abs_senti,positive_matrix)
    sentiment_matrix = np.concatenate([negative_matrix, positive_matrix], axis=1)
    
    read_pca = pickle.load(open(path_in + "pca.pkl","rb"))
    my_pca_matrix = np.array(read_pca.transform(sentiment_matrix))

    tweets = df.full_text
    tweets = [x if type(x) == str else "None" for x in tweets]

    print('extracting BERT embeddings...')
    BERT = bert_features(tweets, cuda=True)

    print('identifying hate features...')
    HATE = hate_features(tweets)
    
    X = np.concatenate([BERT, HATE, my_pca_matrix], axis=1)
    
    read_optimal_model = pickle.load(open(path_in + "logistic_regression.pkl","rb"))
    
    my_prediction = read_optimal_model.predict_proba(X)
    my_result = pd.DataFrame(my_prediction)
    my_result.columns = read_optimal_model.classes_
    max = my_result.max(axis=1)
    my_predict = read_optimal_model.predict(X)
    df['Step 2 Prob'] = max
    df['Step 2 Result'] = my_predict
    
    df.to_csv(out_fn, index=False)
    
    