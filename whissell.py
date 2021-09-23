#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:02:42 2021

@author: ericchen
"""

# Importing and cleaning data for testing
import pandas as pd
import dask
import random
import re

def label_whissel(df, column, whissel):
    from nltk.stem import PorterStemmer
    import numpy as np
    my_stemmer = PorterStemmer()
    
    df['whissel_pls'] = np.nan
    df['whissel_act'] = np.nan
    df['whissel_img'] = np.nan
    
    i = 0
    total_words = 0
    matched = 0

    for index, row in df.iterrows():
        i = i+1
        act = 0
        pls = 0
        img = 0
        num_words = 0
        tweet = row[column]
        processed_tweet = process_tweet(tweet)
        words = processed_tweet.split()
        
        if i %100 == 0: 
            print(i)
        
        for word in words:
            if word in whissel.word.values:
                index1 = whissel.word[whissel.word == word].index[0]
                act = act + whissel.at[index1, 'Act']
                pls = pls + whissel.at[index1, 'Pls']
                img = img + whissel.at[index1, 'Img']
                num_words = num_words +1
                total_words +=1
                matched +=1
            else: 
                stemmed = my_stemmer.stem(word)
                if stemmed in whissel.word.values: 
                    index1 = whissel.word[whissel.word == stemmed].index[0]
                    act = act + whissel.at[index1, 'Act']
                    pls = pls + whissel.at[index1, 'Pls']
                    img = img + whissel.at[index1, 'Img']
                    num_words = num_words +1
                    total_words+=1
                    matched +=1
                else:
                    total_words +=1
        
        if (not num_words == 0): 
            df.at[index, 'whissel_pls'] = pls/num_words
            df.at[index, 'whissel_act'] = act/num_words
            df.at[index, 'whissel_img'] = img/num_words
        
    return df, matched/total_words
        

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
    #remove certain symbols
    tweet = re.sub('=', ' equals ', tweet)
    #trim
    tweet = tweet.strip('\'"')

    return tweet.lower().strip()

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

def average(df):
    return [df['whissel_pls'].mean(),df['whissel_act'].mean(), df['whissel_img'].mean()]
    
#%%

folder_path = '/Users/ericchen/Desktop/CIS-Research/new/'
whissel = pd.read_csv(folder_path + 'dict_of_affect.csv' ,header=0)
whissel.columns = ['word', 'Pls', 'Act', 'Img']
hostile = pd.read_csv(folder_path + 'hostile.csv', header=0)
criticism = pd.read_csv(folder_path + 'criticism.csv', header=0)
counterhate = pd.read_csv(folder_path + 'counterhate.csv', header=0)
neutral =pd.read_csv(folder_path + 'neutral.csv', header=0)

#%%
hostile, match_rate = label_whissel(hostile, "full_text", whissel)
print(match_rate)
criticism, match_rate = label_whissel(criticism, "full_text", whissel)
print(average(criticism))
counterhate = label_whissel(counterhate, "full_text", whissel)
neutral,match_rate = label_whissel(neutral, "full_text", whissel)
print(match_rate)
print(average(neutral))

#%%
corpora = pd.read_csv(folder_path + 'hate.csv' ,header=0)
hostile = pd.read_csv(folder_path + 'hostile.csv', header=0)
criticism = pd.read_csv(folder_path + 'criticism.csv', header=0)

def label_full (corpus, df):
    df["full_text"] = ""
    i = 0
    for index, row in df.iterrows():
        i = i+1
        if i %100 == 0: 
            print(i)
        df.at[index,'full_text']= corpus.loc[corpus['Unnamed: 0']==row["Tweet_Num"]]['full_text'].tolist()[0]
    return df
    
criticism = label_full(corpora,criticism)

hostile = label_full(corpora,hostile)


        
#%%
hostile, match_rate1 = label_whissel(hostile, "full_text", whissel)
print(match_rate1)

criticism, match_rate2 = label_whissel(criticism, "full_text", whissel)
print(match_rate2)

neutral, match_rate3 = label_whissel(neutral, "full_text", whissel)
print(match_rate3)

#%%
counterhate, match_rate4 = label_whissel(counterhate, "full_text", whissel)
print(match_rate4)

#%%
print(average(hostile))
print(average(criticism))
print(average(neutral))
print(average(counterhate))

#%%
def z_score(average):
    import numpy as np
    mean = np.array([1.81, 1.85, 1.94])
    std = np.array([0.44,0.39,0.63])
    average = np.asarray(average)
    
    return (average-mean)/std

z1 = z_score(average(hostile))
z2 =z_score(average(criticism))
z3 = z_score(average(neutral))
z4 = z_score(average(counterhate))

#%%
def stdev(df):
    import numpy as np
    return np.array([df['whissel_pls'].mean(),df['whissel_act'].mean(), df['whissel_img'].mean()])

def z_test(df1, df2):
    import numpy as np
    average1 = np.asarray(average(df1))
    average2 = np.asarray(average(df2))
    
    std1 = stdev(df1)
    std2 = stdev(df2)
    return (average1-average2)/(std1**2/19722+std2**2/7483)**(1/2)

z_test(hostile, criticism)

h = hostile.dropna(axis=0)
c = criticism.dropna(axis=0)
    



        
