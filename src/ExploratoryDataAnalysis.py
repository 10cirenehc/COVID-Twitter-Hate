 # -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:46:17 2020

@author: Eric Chen
"""


# Importing and cleaning data for testing
import pandas as pd
import json_lines as jsl
import dask
import random
import re

random.seed(10)

df = pd.DataFrame()
kept_columns = ['created_at', 'id', 'full_text', 'retweet_count']

json_list = []
###for i in range(1,9):
i = 5
delete_columns = []
j = 0
with open('all_tweet_pt'+str(i)+'.jsonl', 'rb') as f: 
    for item in jsl.reader(f, broken=True):
        
        j = j+1
        num = random.random()
        if num < 0.05: 
            if item['retweet_count'] > 100:
        if j%10000 == 0:
            print(j)
            
    temp = pd.DataFrame.from_records(json_list)
    for col in temp.columns:
        if col not in kept_columns:
            delete_columns.append(col)
    temp= temp.drop(columns = delete_columns)
    temp = temp.drop_duplicates('id')      
    json_list.clear()
    df = df.append(temp, ignore_index = True)
    del temp

df['created_at'] =pd.to_datetime(df.created_at, utc=True)
df = df.sort_values(by=['created_at'])
df_retweet = df.sort_values(by=['retweet_count'], ascending = False)

df = df.drop_duplicates('id')

print(len(df['id'].unique()))

df.tail()

df.to_csv('final_data/tweet_dataset.csv')

count=0

# Replacing id with id_str
for i in range(8,11):
    j = 0
    with open('all_tweet_pt'+str(i)+'.jsonl','rb') as f: 
        for item in jsl.reader(f, broken=True):
            j = j+1
            if item.get('id') in classify_results.id.values:
                index = classify_results.id[classify_results.id == item.get('id')].index[0]
                classify_results.at[index, 'id'] = item.get('id_str')
                count = count+1
            if j%10000 == 0:
                print(j)
            if count%100 == 0: 
                print("Count = " + str(count))
        f.close()
        
        
                
        
#Marking with Ziems' classifier
import numpy as np
classify_results = pd.read_csv('tweet_dataset_out.csv', header= 0)
classify_results['ziems_result'] = ' '
ziems_classifier = pd.read_csv('all_classifications.csv')
ziems_classifier.columns = ['Tweet_ID', 'Label']

j=0
for index, row in classify_results.iterrows():
    j = j+1
    if j% 100:
        print(j)
    if row.id in ziems_classifier.Tweet_ID.values:
        index1 = ziems_classifier.Tweet_ID[ziems_classifier.Tweet_ID == row.id].index[0]
        classify_results.at[index, 'ziems_result'] = ziems_classifier.at[index1, 'Label']



# Counterhate tweets
#print(df.loc[df['id'] == 1217273043354964000].full_text[0])
all_counterhate = pd.read_csv('counterhate.csv' , header = 0)
all_counterhate.columns = ['Tweet_ID', 'User_ID', 'prob1', 'prob2', 'prob3', 'Label']

json_list = []
for i in range(8,10): 
    j = 0
    with open('all_tweet_pt'+str(i)+'.jsonl','rb') as f: 
        for item in jsl.reader(f, broken=True):
            j = j+1
            if item.get('id') in all_counterhate.Tweet_ID.values:
                json_list.append(item)
                print('x')
            if j%10000 ==0:
                print(j)
                
counterhate = pd.DataFrame.from_records(json_list)
for col in counterhate.columns:
    if col not in kept_columns:
        delete_columns.append(col)
counterhate= counterhate.drop(columns = delete_columns)
counterhate = counterhate.drop_duplicates('id')   

counterhate.to_csv('final_data/counterhate.csv')
classify_results.to_csv('final_data/tweet_dataset_out1.csv')



# Explortory analysis of annotated datasets
import matplotlib as plt
import numpy as np

vidgen = pd.read_csv('vidgen.tsv', header = 0, sep = '\t')

with open('annotations_summary.csv') as f: 
    print(f.encoding)
ziems = pd.read_csv('annotations_summary.csv', header = 0, encoding = 'unicode_escape')


def clean_text(var):
    import re
    tmp = re.sub('[^A-z]+', ' ', var).lower()
    return tmp

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
    #remove certain symbols
    tweet = re.sub('=', ' equals ', tweet)
    #trim
    tweet = tweet.strip('\'"')

    return tweet.lower().strip()

#Labelling datasets with whissell
def label_whissel(df, column, whissel):
    from nltk.stem import PorterStemmer
    import numpy as np
    my_stemmer = PorterStemmer()
    
    df['whissel_pls'] = np.nan
    df['whissel_act'] = np.nan
    df['whissel_img'] = np.nan
    
    i = 0
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
            else: 
                stemmed = my_stemmer.stem(word)
                if stemmed in whissel.word.values: 
                    index1 = whissel.word[whissel.word == stemmed].index[0]
                    act = act + whissel.at[index1, 'Act']
                    pls = pls + whissel.at[index1, 'Pls']
                    img = img + whissel.at[index1, 'Img']
                    num_words = num_words +1
        
        if ( not num_words == 0): 
            df.at[index, 'whissel_pls'] = pls/num_words
            df.at[index, 'whissel_act'] = act/num_words
            df.at[index, 'whissel_img'] = img/num_words
        
    return df
        

import datetime
whissel = pd.read_csv('final_data/dict_of_affect.csv' ,header=0)
whissel.columns = ['word', 'Pls', 'Act', 'Img']
ziems_annotated = pd.read_csv('annotations.csv' , header = 0)
vidgen_annotated = pd.read_csv('vidgen.tsv', header = 0, sep = '\t')
classified = pd.read_csv('final_data/tweet_dataset_out.csv', header = 0)

ziems_annotated = label_whissel(ziems_annotated, 'Text', whissel)
vidgen_annotated = label_whissel(vidgen_annotated, 'text', whissel)
classified = label_whissel(classified, 'full_text', whissel)
print (classified.at[0, 'full_text'])

test_string = '@getongab zoom=chink trash'
processed_tweet = process_tweet(test_string)
print(processed_tweet)

classified.to_csv('test_with_whissel.csv')
vidgen_annotated.to_csv('vidgen_with_whissel.csv')
ziems_annotated.to_csv('ziems_with_whissel.csv')

counterhate = pd.read_csv('final_data/counterhate.csv')
counterhate = label_whissel(counterhate, 'full_text', whissel)
columns = ['Prob', 'Result']

classified = classified.drop(columns = columns)
classified = classified.append(counterhate, ignore_index = True)

classified['date'] = pd.to_datetime(classified['created_at']).dt.date
classified = classified.sort_values(by = 'date')

s = pd.to_datetime(classified['date'])
time_data = s.groupby(s.dt.floor('d')).size().reset_index(name='count')
time_data['date'] = pd.to_datetime(time_data['date']).dt.date



import matplotlib.pyplot as plt
import numpy as np

labels = ['Hate', 'Neutral', 'Counter Hate']
sizes = [38, 50, 12]
colors = ['0.25', '0.5', '0.75']
plt.pie(sizes, labels = labels, autopct = '%1.0f%%', shadow = False, startangle = 128, colors = colors)
plt.title('Distribution of Tweets in Ziems et al. Annotated Dataset\n')
plt.axis('equal')
plt.show()


x = time_data.date.tolist()
y = time_data['count'].tolist()
plt.plot(x,y, color = '0.7',linewidth = 1, marker = '^', markerfacecolor = '0.2', markersize = 1)
plt.xlabel('Date (2020)')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Tweets in sampled dataset over time')
plt.show()

none_of_the_above = 0
entity_directed_criticism = 0
entity_directed_hostility = 0
discussion_of_eastasian_prejudice = 0
counter_speech = 0

for index, row in vidgen_annotated.iterrows():
    if row.expert == 'none_of_the_above':
        none_of_the_above = none_of_the_above+1
    elif row.expert == 'entity_directed_criticism':
        entity_directed_criticism = entity_directed_criticism +1
    elif row.expert == 'entity_directed_hostility':
        entity_directed_hostility = entity_directed_hostility+1
    elif row.expert == 'discussion_of_eastasian_prejudice':
        discussion_of_eastasian_prejudice = discussion_of_eastasian_prejudice+1
    elif row.expert =='counter_speech':
        counter_speech = counter_speech+1

labels = ('Entity Directed Hostility', 'Entity Directed Criticism','Discussion of East Asian Prejudice', 'None of the above', 'Counter Speech')
sizes = [3898, 1433, 1029, 13524, 116]
colors = ['0.25', '0.4', '0.55', '0.7', '0.85']
plt.pie(sizes, labels = labels, autopct = '%1.0f%%', shadow = False, startangle = 128, colors = colors)
plt.title('Distribution of Tweets in Vidgen et al. Annotated Dataset\n')
plt.axis('equal')
plt.show()

import pandas as pd
results = pd.read_csv('project/tweet_dataset_out.csv')
















                
                
            
        
        
        
    
    


