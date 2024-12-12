import numpy as np
import pandas as pd
import nltk
# nltk.download('all')
from collections import Counter

from keras.api.layers import Dense, Embedding, Input, InputLayer, RNN, SimpleRNN, LSTM, Bidirectional, TimeDistributed,TextVectorization
from keras.api.models import Model, Sequential
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from keras.api.preprocessing.sequence import pad_sequences
import tensorflow as tf

import string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from sklearn.model_selection import train_test_split

import spacy
from time import time
import numpy as np

import re

#these include the hyperparameters also
config = {'min_text_len':30,
          'max_text_len':63,
          'max_summary_len':30,
          'latent_dim' : 300,
          'embedding_dim' : 200}


def load_data():
    summary = pd.read_csv('artifacts/news_summary.csv', encoding='iso-8859-1')
    raw = pd.read_csv('artifacts/news_raw.csv', encoding='iso-8859-1')
    raw = raw.rename(columns = {'headlines':'summary'})
    summary = summary[['headlines', 'text']].rename(columns={'headlines':'summary'})

    df = pd.concat([raw, summary]).reset_index(drop=True)
    print(df.shape)
    print(summary.shape, raw.shape)

    return df,raw

def pre_process_data():
    df,raw = load_data()
    print(f'Before filtering: {raw.shape}')
    pre = df.loc[((df['text'].str.split(" ").str.len()>config['min_text_len'])
                &(df['text'].str.split(" ").str.len()<config['max_text_len']))].reset_index(drop=True)
    print(f'After filtering: {pre.shape}')

    return pre

def text_strip(sentence):
    # Remove non-alphabetic characters (Data Cleaning)

    sentence = re.sub("(\\t)", " ", str(sentence)).lower()
    sentence = re.sub("(\\r)", " ", str(sentence)).lower()
    sentence = re.sub("(\\n)", " ", str(sentence)).lower()

    # Remove - if it occurs more than one time consecutively
    sentence = re.sub("(--+)", " ", str(sentence)).lower()

    # Remove . if it occurs more than one time consecutively
    sentence = re.sub("(\.\.+)", " ", str(sentence)).lower()

    # Remove the characters - <>()|&©ø"',;?~*!
    sentence = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", " ", str(sentence)).lower()

    # Remove \x9* in text
    sentence = re.sub(r"(\\x9\d)", " ", str(sentence)).lower()

    # Replace CM# and CHG# to CM_NUM
    sentence = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", "CM_NUM", str(sentence)).lower()

    # Remove punctuations at the end of a word
    sentence = re.sub("(\.\s+)", " ", str(sentence)).lower()
    sentence = re.sub("(\-\s+)", " ", str(sentence)).lower()
    sentence = re.sub("(\:\s+)", " ", str(sentence)).lower()

    # Remove multiple spaces
    sentence = re.sub("(\s+)", " ", str(sentence)).lower()

    return sentence

def getCleanData():
    pre = pre_process_data()
    pre['cleaned_text'] = pre['text'].apply(lambda x: text_strip(x))
    #why is sostok and eostok added despite adding start and end token?
    #these token are not required on the long text side(encode side)
    pre['cleaned_summary'] = pre['summary'].apply(lambda x: '_START_ '+ text_strip(x) + ' _END_')
    pre['cleaned_summary'] = pre['cleaned_summary'].apply(lambda x: 'sostok ' + x + ' eostok')


    #remove below filter if required=>adding to restrict size of data
    post_pre = pre[((pre.cleaned_text.str.split().str.len()<=config['max_text_len']) &
                    (pre.summary.str.split().str.len()<=(config['max_summary_len']+4)))].copy()
    post_pre = post_pre.reset_index(drop=True)
    print(post_pre.shape)

    post_pre = post_pre.drop(['text', 'summary'], axis=1)
    post_pre = post_pre.rename(columns = {'cleaned_text':'text',
                                        'cleaned_summary':'summary'})
    print(post_pre.head())

    return post_pre

def splitData():
    post_pre = getCleanData()
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(post_pre["text"]),
                                            np.array(post_pre["summary"]),
                                            test_size=0.1,
                                            random_state=0,
                                            shuffle=True
                                           )

    print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

    return x_train, x_valid, y_train, y_valid

def get_rare_words(text_col, thresh=5):
    # Convert the text column to a list of strings
    text_list = list(text_col)

    # Initialize the TextVectorization layer
    text_vectorizer = tf.keras.layers.TextVectorization(output_mode='int')
    text_vectorizer.adapt(text_list)

    # Get the vocabulary from the TextVectorization layer
    vocab = text_vectorizer.get_vocabulary()  # Index-to-word mapping

    # Tokenize the text data
    tokenized_texts = text_vectorizer(text_list)

    # Flatten the tokenized output and count occurrences of each token
    token_counts = Counter()
    for tokens in tokenized_texts:
        for token in tokens.numpy():
            if token != 0:  # Ignore padding (0)
                word = vocab[token]
                token_counts[word] += 1

    # Count rare words based on the threshold
    tot_cnt = len(token_counts)
    cnt = sum(1 for count in token_counts.values() if count < thresh)

    print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)

    return cnt, tot_cnt

def tokenize_train_and_validation_dataset():
    x_train, x_valid, y_train, y_valid = splitData()
    x_train_cnt, x_train_tot_cnt = get_rare_words(text_col=x_train)

    xtext_vectorizer = TextVectorization(output_mode='int', 
                                    output_sequence_length=config['max_text_len'], 
                                    max_tokens=x_train_tot_cnt - x_train_cnt)

    # Adapt the vectorizer to the training data
    xtext_vectorizer.adapt(x_train)

    # Transform training and validation texts into integer sequences
    x_tr = xtext_vectorizer(x_train)
    x_val = xtext_vectorizer(x_valid)

    # Size of vocabulary (+1 for padding token)
    x_voc = len(xtext_vectorizer.get_vocabulary()) + 1

    print("Size of vocabulary in X = {}".format(x_voc))

    y_train_cnt, y_train_tot_cnt = get_rare_words(text_col=y_train)
    
    ytext_vectorizer = TextVectorization(output_mode='int', 
                                    output_sequence_length=config['max_text_len'], 
                                    max_tokens=y_train_tot_cnt - y_train_cnt)

    # Adapt the vectorizer to the val data
    ytext_vectorizer.adapt(x_train)

    y_tr = ytext_vectorizer(x_train)
    y_val = ytext_vectorizer(x_valid)

    # Size of vocabulary (+1 for padding token)
    y_voc = len(ytext_vectorizer.get_vocabulary()) + 1

    print("Size of vocabulary in X = {}".format(y_voc))



if __name__=="__main__":
    tokenize_train_and_validation_dataset()