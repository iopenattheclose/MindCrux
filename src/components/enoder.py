from keras.api.layers import Dense, Embedding, Input, InputLayer, RNN, SimpleRNN, LSTM, Bidirectional, TimeDistributed,TextVectorization
from keras.api.models import Model, Sequential
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import tokenize_train_and_validation_dataset


#these include the hyperparameters also
config = {'min_text_len':30,
          'max_text_len':63,
          'max_summary_len':30,
          'latent_dim' : 300,
          'embedding_dim' : 200}

def initialize_encode_architectire():
    latent_dim = config['latent_dim']
    embedding_dim = config['embedding_dim']
    max_text_len = config['max_text_len']
    max_summary_len = config['max_summary_len']

    x_voc,y_voc = tokenize_train_and_validation_dataset()

    # Encoder
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
    decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.summary()