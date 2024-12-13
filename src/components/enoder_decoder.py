from keras.api.layers import Dense, Embedding, Input, InputLayer, RNN, SimpleRNN, LSTM, Bidirectional, TimeDistributed,TextVectorization
from keras.api.models import Model, Sequential
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import tokenize_train_and_validation_dataset
import tensorflow as tf
import os
import sys
sys.path.append("src")
from utils import save_object
import numpy as np


#these include the hyperparameters also
config = {'min_text_len':30,
          'max_text_len':63,
          'max_summary_len':30,
          'latent_dim' : 300, #ht vec,ct vec, it vec, ft vec, ot vec
          'embedding_dim' : 200}

def initialize_encoder_decoder_architecture():
    latent_dim = config['latent_dim']
    embedding_dim = config['embedding_dim']
    max_text_len = config['max_text_len']
    max_summary_len = config['max_summary_len']

    x_voc,y_voc,x_tr,y_tr,x_val,y_val = tokenize_train_and_validation_dataset()

    # Encoder input sequence(long text length)
    encoder_inputs = Input(shape=(max_text_len, ))

    # Embedding layer
    enc_emb = Embedding(x_voc, embedding_dim,
                        trainable=True)(encoder_inputs)

    # Encoder LSTM 1
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                        return_state=True, dropout=0.4,
                        recurrent_dropout=0.4)
    (encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)

    # Encoder LSTM 2
    encoder_lstm2 = LSTM(latent_dim, return_sequences=True,
                        return_state=True, dropout=0.4,
                        recurrent_dropout=0.4)
    (encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)

    # Encoder LSTM 3
    encoder_lstm3 = LSTM(latent_dim, return_state=True,
                        return_sequences=True, dropout=0.4,
                        recurrent_dropout=0.4)
    (encoder_outputs, state_h, state_c) = encoder_lstm3(encoder_output2)

    # Set up the decoder, using encoder_states as the initial state
    decoder_inputs = Input(shape=(None, ))

    # Embedding layer
    dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # Decoder LSTM
    decoder_lstm = LSTM(latent_dim, return_sequences=True,
                        return_state=True, dropout=0.4,
                        recurrent_dropout=0.2)
    (decoder_outputs, decoder_fwd_state, decoder_back_state) = \
        decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    # Dense layer
    #TimeDistributed is used to predict which word out of v is to be predicted at each output of the LSTM return state(time step )
    decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    print(model.summary())

    return model,np.array(x_tr),np.array(y_tr),np.array(x_val),np.array(y_val)

def train_and_save_model():

    model,x_tr,y_tr,x_val,y_val = initialize_encoder_decoder_architecture()
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_name = "./weights.weights.h5"
    save_model = ModelCheckpoint(filepath=model_name,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)    

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    save_object(file_path=os.path.join("artifacts","model.pkl"),obj = model)

    history = model.fit(
    [x_tr, y_tr[:, :-1]],
    y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],
    epochs=500,
    callbacks=[es, save_model],
    batch_size=1024,
    validation_data=([x_val, y_val[:, :-1]],
                     y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]),
    )





if __name__=="__main__":
    train_and_save_model()