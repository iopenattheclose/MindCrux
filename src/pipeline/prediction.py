import os
import sys
sys.path.append("src/components")
from src.components.data_preprocessing import splitData,get_rare_words
from keras.api.layers import Dense, Embedding, Input, InputLayer, RNN, SimpleRNN, LSTM, Bidirectional, TimeDistributed,TextVectorization
from utils import *
from src.components.enoder_decoder import initialize_encoder_decoder_architecture,tokenize_train_and_validation_dataset
from keras.api.models import Model, Sequential


def convertIndexToWords():
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

    source_vocab = xtext_vectorizer.get_vocabulary()  # Returns list of source tokens
    target_vocab = ytext_vectorizer.get_vocabulary()  # Returns list of target tokens

    # Create reverse word index mappings (index -> word)
    reverse_source_word_index = {i: word for i, word in enumerate(source_vocab)}
    reverse_target_word_index = {i: word for i, word in enumerate(target_vocab)}

    # Create word-to-index mappings (word -> index) for the target vocabulary
    target_word_index = {word: i for i, word in enumerate(target_vocab)}

    return reverse_source_word_index,reverse_target_word_index,target_word_index

def decoder_inference_architecture():
    latent_dim = config['latent_dim']
    embedding_dim = config['embedding_dim']
    max_text_len = config['max_text_len']
    max_summary_len = config['max_summary_len']

    #using only xvoc
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


    #############
    # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs,
                        state_h, state_c])

    # Decoder setup

    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim, ))
    decoder_state_input_c = Input(shape=(latent_dim, ))
    decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

    # Get the embeddings of the decoder sequence
    #dec_emb_layer is already trained
    dec_emb2 = dec_emb_layer(decoder_inputs)

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    (decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
            initial_state=[decoder_state_input_h, decoder_state_input_c])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,
                        decoder_state_input_h, decoder_state_input_c],
                        [decoder_outputs2] + [state_h2, state_c2])