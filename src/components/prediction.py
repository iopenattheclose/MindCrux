import os
import sys
from data_preprocessing import splitData,get_rare_words
from keras.api.layers import Dense, Embedding, Input, InputLayer, RNN, SimpleRNN, LSTM, Bidirectional, TimeDistributed,TextVectorization
from enoder_decoder import initialize_encoder_decoder_architecture,tokenize_train_and_validation_dataset
from keras.api.models import Model, Sequential
import keras
from utils import config
import numpy as np
import pandas as pd

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

    return reverse_source_word_index,reverse_target_word_index,target_word_index,x_tr,y_tr

def decoder_inference_architecture():
    latent_dim = config['latent_dim']
    embedding_dim = config['embedding_dim']
    max_text_len = config['max_text_len']
    max_summary_len = config['max_summary_len']

    loaded_model = keras.models.load_model("artifacts/my_model.keras")

    # 1. Get encoder_inputs and encoder_outputs:
    encoder_inputs = loaded_model.input[0]  # InputLayer name=input_layer
    encoder_outputs, state_h, state_c = loaded_model.get_layer('lstm_3').output  # Outputs from LSTM name=lstm_3

    # 2. Recreate encoder_model:
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])


    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

    dec_emb_layer = loaded_model.get_layer('embedding_1')  # Embedding name=embedding_1
    decoder_inputs = Input(shape=(None,))  # Define decoder_inputs
    dec_emb2 = dec_emb_layer(decoder_inputs)

    decoder_lstm = loaded_model.get_layer('lstm_2')  # LSTM name=lstm_2
    (decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    decoder_dense = loaded_model.get_layer('time_distributed')  # TimeDistributed name=time_distributed
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # 4. Recreate decoder_model:
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2]
    )

    return encoder_model, decoder_model


def decode_sequence(input_seq):

    reverse_source_word_index,reverse_target_word_index,target_word_index = convertIndexToWords()
    encoder_model, decoder_model = decoder_inference_architecture()
    # Encode the input as state vectors.
    (e_out, e_h, e_c) = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq]
                + [e_out, e_h, e_c], verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the stop word.
        if sampled_token == 'eostok' or len(decoded_sentence.split()) \
            >= config['max_summary_len'] - 1:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        (e_h, e_c) = (h, c)

    return decoded_sentence

# To convert sequence to text
def seq2text(input_seq):
    reverse_source_word_index,reverse_target_word_index,target_word_index,x_tr,y_tr = convertIndexToWords()
    newString = ''
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + ' '

    return newString

# To convert sequence to summary
def seq2summary(input_seq):
    reverse_source_word_index,reverse_target_word_index,target_word_index,x_tr,y_tr = convertIndexToWords()
    newString = ''
    for i in input_seq:
        if (i != 0) and (i != target_word_index['sostok']) and (i != target_word_index['eostok']):
            newString = newString + reverse_target_word_index[i] + ' '

    return newString

def final_predict():
    reverse_source_word_index,reverse_target_word_index,target_word_index,x_tr,y_tr = convertIndexToWords()
    x_tr,y_tr = np.array(x_tr),np.array(y_tr)

    actual = []
    predicted = []
    for i in range(0, 3):
        print ('Review:', seq2text(x_tr[i]))

        actual.append(seq2summary(y_tr[i]))
        print ('Original summary:', actual[-1])

        predicted.append(decode_sequence(x_tr[i].reshape(1, config['max_text_len'])))
        print ('Predicted summary:', predicted[-1])
        print()

    prediction_df = pd.DataFrame({'Actual':actual, 'Predicted':predicted})
    prediction_df.head(10)

if __name__=="__main__":
    final_predict()