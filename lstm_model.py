import os
import pickle
import time

import keras.callbacks
import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.layers import LSTM, Embedding
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from utils import TRUTH_LABELS, COMMENT_TEXT_INDEX
from utils import transform_text_in_df_return_w2v_np_vectors

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

MAX_BATCH_SIZE_PRE_TRAINED = 400
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

MAX_NUM_WORDS_ONE_HOT = 50000

FILE_NAME_STRING_DELIMITER = "_"
FILE_NAME_STRING_FORMATING = "%d_%m_%y_%H:%M"
KERAS_MODEL_DIRECTORY = 'keras_models/{}'
TRAIN_HISTORY_DICT_PATH = 'keras_models/{}/trainHistoryDict'
MODEL_SAVE_PATH = 'keras_models/{}/keras_model.h5'

MAXLEN = 100


def lstm_main(summarized_sentences, truth_dictionary, w2v_model, testing, use_w2v=True):
    if testing:
        print("running tests")
        number_of_epochs = 1
    else:
        print("running eval")
        number_of_epochs = 1

    # process data
    print("processing data")
    if use_w2v:
        np_vector_array = transform_text_in_df_return_w2v_np_vectors(summarized_sentences, w2v_model)

        model_dict = {}
        results_dict = {}

        for key in TRUTH_LABELS:
            x_train, x_test, y_train, y_test = train_test_split(np_vector_array, truth_dictionary[key],
                                                                test_size=0.1,
                                                                random_state=42)
            x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)

            model = build_keras_model(max_len=MAXLEN)
            print("training network")

            for e in range(number_of_epochs):
                print("epoch %d" % e)
                for X_train, Y_train in batch_generator(x_train, y_train):
                    model.fit(X_train, Y_train, batch_size=200, nb_epoch=1)

            validation = model.predict_classes(x_test)
            print('\nConfusion matrix\n', confusion_matrix(y_test, validation))
            print(classification_report(y_test, validation))
            model_dict[key] = model
            results_dict[key] = validation
            # try some values
        return model_dict, results_dict
    else:

        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_ONE_HOT,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=" ",
                              char_level=False)
        tokenizer.fit_on_texts(summarized_sentences)
        transformed_text = tokenizer.texts_to_sequences(summarized_sentences)
        vocab_size = len(tokenizer.word_counts)

        print("vocab length is", len(tokenizer.word_counts))
        model_dict = {}
        results_dict = {}
        for key in TRUTH_LABELS:
            x_train, x_test, y_train, y_test = train_test_split(transformed_text, truth_dictionary[key],
                                                                test_size=0.1,
                                                                random_state=42)
            x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)


            # build neural network model
            print("training network")
            model = build_keras_embeddings_model(max_size=vocab_size, max_length=MAXLEN)

            for e in range(number_of_epochs):
                print("epoch %d" % e)
                for X_train, Y_train in batch_generator(x_train, y_train):
                    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

            validation = model.predict_classes(x_test)
            print('\nConfusion matrix\n', confusion_matrix(y_test, validation))
            print(classification_report(y_test, validation))
            model_dict[key] = model
            results_dict[key] = validation
        return model_dict, results_dict, tokenizer  # THIS IS FAKE


def lstm_predict(model_dict, tokenizer, predicted_data, truth_dictionary, w2v_model, use_w2v=True):
    if use_w2v:
        prediction_sentences = predicted_data[COMMENT_TEXT_INDEX]
        np_text_array = transform_text_in_df_return_w2v_np_vectors(prediction_sentences, w2v_model)
        padded_x_test = sequence.pad_sequences(np_text_array, maxlen=MAXLEN)
        results_dict = {}
        for key in truth_dictionary:
            model = model_dict[key]
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(index=-2).output)
            intermediate_output = intermediate_layer_model.predict(padded_x_test)
            results_dict[key] = np.array(intermediate_output)
    else:
        prediction_sentences = predicted_data[COMMENT_TEXT_INDEX]
        tokenized_predictions = tokenizer.texts_to_sequences(prediction_sentences)
        padded_x_test = sequence.pad_sequences(tokenized_predictions, maxlen=MAXLEN)
        results_dict = {}
        for key in truth_dictionary:
            model = model_dict[key]
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(index=-2).output)
            intermediate_output = intermediate_layer_model.predict(padded_x_test)
            results_dict[key] = np.array(intermediate_output)
    return results_dict


def batch_generator(x_train, y_train):
    i = 1000
    while i < len(x_train) + 1000:
        x = sequence.pad_sequences(x_train[i - 1000:i], maxlen=MAXLEN)
        y = y_train[i - 1000:i]
        print(x.shape)
        yield x, y
        i += 1000


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime(FILE_NAME_STRING_FORMATING) + FILE_NAME_STRING_DELIMITER + expt_name
    os.makedirs(KERAS_MODEL_DIRECTORY.format(folder), exist_ok=True)
    model.save(MODEL_SAVE_PATH.format(folder))
    with open(TRAIN_HISTORY_DICT_PATH.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def build_keras_model(max_len, testing=False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(max_len, 300)))
    if not testing:
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def build_keras_embeddings_model(max_size, max_length, testing=False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Embedding(max_size, 64, input_length=max_length))
    if not testing:
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
