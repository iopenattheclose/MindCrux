import numpy as np
import pandas as pd
import nltk
# nltk.download('all')

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

    ind = 44
    print(f'Text: {pre.text[ind]}')
    print()
    print(f'Summary: {pre.summary[ind]}')
    print()
    print(f'Text length: {len(pre.text[ind].split())}')
    print(f'Summary length: {len(pre.summary[ind].split())}')




if __name__=="__main__":
    pre_process_data()