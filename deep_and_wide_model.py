import os
import pickle
import time

import keras.backend as K
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Merge
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from gazette_model import process_bad_words
from lda_model import get_lda_topics
from lsi_model import build_LSI_model
from tf_idf_model import tf_idf_vectorizer_small, tf_idf_vectorizer_big, build_logistic_regression_model
from utils import COMMENT_TEXT_INDEX, TRUTH_LABELS
from utils import load_data, extract_truth_labels_as_dict, \
    transform_text_in_df_return_w2v_np_vectors, load_w2v_model_from_path

IGNORE_FLAG = 0
TRAIN_NEW_FLAG = 1
USE_OLD_FLAG = 2

FAST_TEXT_FLAG = "fast_text"
TF_IDF_FLAG = "tf-idf"
LSI_FLAG = "lsi"
LDA_FLAG = "lda"
NOVEL_FLAG = "lstm_novel"
W2V_FLAG = "lstm_w2v"
GAZETTE_FLAG = "gazette"

SUM_SENTENCES_FILE = './data/newtrain.p'
FILE_NAME_STRING_DELIMITER = "_"
FILE_NAME_STRING_FORMATING = "%d_%m_%y_%H_%M_%S"
KERAS_MODEL_DIRECTORY = 'keras_models/{}'
TRAIN_HISTORY_DICT_PATH = 'keras_models/{}/trainHistoryDict'
MODEL_SAVE_PATH = 'keras_models/{}/keras_model.h5'

SPARSE_ARRAY_NAME = "sparse_array.npy"
PRE_TRAINED_RESULT = "pre_train.npy"
NOVEL_TRAINED_RESULT = "novel_train.npy"
TF_IDF_SMALL = "tf_idf_small.npy"
TF_IDF_BIG = "tf_idf_small.npy"
LSI_MODEL = "lsi.npy"
LDA_MODEL = "lda.npy"

BATCH_SIZE = 10

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

MAX_BATCH_SIZE_PRE_TRAINED = 400

MAX_NUM_WORDS_ONE_HOT = 50000

MAX_W2V_LENGTH = 300


def main(train_data_file, predict_data_file, summarized_sentences, w2v_model, testing, save_file_directory="",
         train_new=True, train_flag_dict=None):
    assert type(summarized_sentences) == list
    assert type(summarized_sentences[0]) == str
    train_df = load_data(train_data_file)
    predict_df = load_data(predict_data_file)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(predict_df, pd.DataFrame)

    # get truth dictionary
    truth_dictionary = extract_truth_labels_as_dict(train_df)
    train_sentences = train_df[COMMENT_TEXT_INDEX]

    if testing:
        summarized_sentences = summarized_sentences[:len(train_df)]
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()

    # convert w2v to array
    np_vector_array = transform_text_in_df_return_w2v_np_vectors(summarized_sentences, w2v_model)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_ONE_HOT,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    tokenizer.fit_on_texts(summarized_sentences)
    transformed_text = np.array(tokenizer.texts_to_sequences(summarized_sentences))
    vocab_size = len(tokenizer.word_counts)

    # get gazette matrices
    if train_flag_dict[GAZETTE_FLAG]:
        if train_new:
            sparse_gazette_matrices = process_bad_words(train_sentences)
            np.save(save_file_directory + SPARSE_ARRAY_NAME, sparse_gazette_matrices)
            assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
            del sparse_gazette_matrices

    # get tf-idf vectorizer
    if train_flag_dict[TF_IDF_FLAG]:
        if train_new:
            vector_small = tf_idf_vectorizer_small(train_sentences)
            np.save(save_file_directory + TF_IDF_SMALL, vector_small)
        else:
            vector_small = np.load(save_file_directory + TF_IDF_SMALL)
        # get log regression score from tf-idf(2-6 n gram) log reg
        if train_new:
            vector_big = tf_idf_vectorizer_big(train_sentences)
            aggressively_positive_model_report = build_logistic_regression_model(vector_big, truth_dictionary)
            np.save(save_file_directory + TF_IDF_BIG, vector_big)
        else:
            vector_big = np.load(save_file_directory + TF_IDF_BIG)
            aggressively_positive_model_report = build_logistic_regression_model(vector_big, truth_dictionary)

    # get lsi
    if train_flag_dict[LSI_FLAG]:
        if train_new:
            lsi_topics = build_LSI_model(train_sentences)
            np.save(save_file_directory + LSI_MODEL, lsi_topics)
        else:
            np.load(save_file_directory + LSI_MODEL)

    # get lda
    if train_flag_dict[LDA_FLAG]:
        if train_new:
            lda_topics = get_lda_topics(train_sentences)
            np.save(save_file_directory + LDA_MODEL, lda_topics)
        else:
            np.load(save_file_directory + LDA_MODEL)

    "sparse = {}, w2v_lstm = {}, novel_lstm = {}, tf-idf = {}, lda = {}, lsi = {}"
    sparse_gazette_matrices = np.load(save_file_directory + SPARSE_ARRAY_NAME)
    assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
    for key in aggressively_positive_model_report:
        aggressively_positive_model_report[key] = np.array(
            [i[1] for i in aggressively_positive_model_report[key]]).reshape((50, 1))
    for key in truth_dictionary:
        np_full_array = np.hstack(
            (sparse_gazette_matrices, lsi_topics, lda_topics, aggressively_positive_model_report[key]))
        print(np_full_array.shape)
        deep_and_wide_network(key=key, np_vector_array=np_vector_array, vocab_size=vocab_size,
                              np_full_array=np_full_array,
                              testing=testing, transformed_text=transformed_text,
                              truth_dictionary=truth_dictionary)


def deep_and_wide_network(key, np_full_array, testing, vocab_size, transformed_text,
                          truth_dictionary, np_vector_array):
    # get w2v lstm matrices
    if testing:
        number_of_epochs = 1
    else:
        number_of_epochs = 5
    print(np_full_array.shape[0])
    print(transformed_text.shape[0])
    print(np_vector_array.shape[0])
    print(truth_dictionary[key].shape[0])
    full_x_train, full_x_test, y_train, y_test = train_test_split(np_full_array, truth_dictionary[key],
                                                                  test_size=0.05, random_state=42)
    w2v_x_word, w2v_x_test, novel_x_train, novel_x_test = train_test_split(np_vector_array, transformed_text,
                                                                           test_size=0.05, random_state=42)
    w2v_model = build_keras_model(max_len=MAX_W2V_LENGTH)
    keras_model = build_keras_embeddings_model(max_size=vocab_size, max_length=MAX_NUM_WORDS_ONE_HOT)

    w2v_x_test = sequence.pad_sequences(w2v_x_test, maxlen=MAX_W2V_LENGTH)
    novel_x_test = sequence.pad_sequences(novel_x_test, maxlen=MAX_NUM_WORDS_ONE_HOT)

    sparse_model = Sequential()
    sparse_model.add(Dense(128, input_shape=(np_full_array.shape[1],)))
    full_model = Sequential()
    full_model.add(Merge([sparse_model, w2v_model, keras_model], mode='concat', concat_axis=1))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(100))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(50))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(10))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(1, activation='sigmoid'))
    full_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    for e in range(number_of_epochs):
        print("epoch %d" % e)
        for X_train, Y_train in batch_generator(full_x_train, w2v_x_word, novel_x_train, y_train):
            full_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=1)
        full_model.evaluate([full_x_train, w2v_x_test, novel_x_test], y_test)

    return full_model

"""
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
            x_test = sequence.pad_sequences(x_test, maxlen=MAX_W2V_LENGTH)

            model = build_keras_model(max_len=MAX_W2V_LENGTH)
            print("training network")

            for e in range(number_of_epochs):
                print("epoch %d" % e)
                for X_train, Y_train in batch_generator(x_train, y_train):
                    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=1)

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
            x_test = sequence.pad_sequences(x_test, maxlen=MAX_W2V_LENGTH)

            # build neural network model
            print("training network")
            model = build_keras_embeddings_model(max_size=vocab_size, max_length=MAX_W2V_LENGTH)

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
"""

def batch_generator(x_train, w2v_vector, novel_vector, y_train):
    i = BATCH_SIZE
    while i < len(x_train) + BATCH_SIZE:
        w2v_vector = sequence.pad_sequences(w2v_vector[i - BATCH_SIZE:i], maxlen=MAX_W2V_LENGTH)
        novel_vector = sequence.pad_sequences(novel_vector[i - BATCH_SIZE:i], maxlen=MAX_NUM_WORDS_ONE_HOT)
        x = x_train[i - BATCH_SIZE:i]
        y = y_train[i - BATCH_SIZE:i]
        yield [x, w2v_vector, novel_vector], y
        i += BATCH_SIZE


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


if __name__ == "__main__":
    SUM_SENTENCES_FILE = './data/newtrain.p'
    summarized_sentence_data = pickle.load(open(SUM_SENTENCES_FILE, "rb"))

    SAMPLE_DATA_FILE = './data/sample.csv'
    TRAIN_DATA_FILE = './data/train.csv'
    PREDICT_DATA_FILE = './data/test_predict.csv'

    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    sample_model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)

    # -----------------------------------------------------------------------------------------------------------------
    # SUPER IMPORTANT FLAG

    train_new = True  # True if training new model, else false
    EXPT_NAME = "09_03_18_10_00_49"  # ONLY USED OF train_new = False
    feature_dictionary = {GAZETTE_FLAG: TRAIN_NEW_FLAG,
                          W2V_FLAG: TRAIN_NEW_FLAG,
                          NOVEL_FLAG: TRAIN_NEW_FLAG,
                          LDA_FLAG: TRAIN_NEW_FLAG,
                          LSI_FLAG: TRAIN_NEW_FLAG,
                          TF_IDF_FLAG: TRAIN_NEW_FLAG,
                          FAST_TEXT_FLAG: IGNORE_FLAG}
    # -----------------------------------------------------------------------------------------------------------------

    if train_new:
        EXPT_NAME = time.strftime(FILE_NAME_STRING_FORMATING)
        SAVE_FILE_PATH = "./expt/" + EXPT_NAME + ""
        TEST_SAVE_FILE_PATH = SAVE_FILE_PATH + "_TEST/"
        REAL_SAVE_FILE_PATH = SAVE_FILE_PATH + "_REAL/"
        os.makedirs(TEST_SAVE_FILE_PATH)
        os.makedirs(REAL_SAVE_FILE_PATH)

        print("preparing to train new model")

        print("doing tests")
        main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=True, save_file_directory=TEST_SAVE_FILE_PATH, train_new=True,
             train_flag_dict=feature_dictionary)

        print("starting real training")
        # real_model = load_w2v_model_from_path(W2V_MODEL)
        main(train_data_file=TRAIN_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=True,
             train_flag_dict=feature_dictionary)
    else:
        print("preparing to reuse old model using flags", feature_dictionary)
        SAVE_FILE_PATH = "./expt/" + EXPT_NAME + ""
        TEST_SAVE_FILE_PATH = SAVE_FILE_PATH + "_TEST/"
        REAL_SAVE_FILE_PATH = SAVE_FILE_PATH + "_REAL/"
        try:
            assert os.path.exists(TEST_SAVE_FILE_PATH)
            assert os.path.exists(REAL_SAVE_FILE_PATH)
        except:
            raise Exception("Experiment path doesn't exist")

        print("doing tests")
        main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=True, save_file_directory=TEST_SAVE_FILE_PATH, train_new=False,
             train_flag_dict=feature_dictionary)

        """
        print("starting expt")
        real_model = load_w2v_model_from_path(W2V_MODEL)  # doing this at the end cause very slow
        main(train_data_file=TRAIN_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=real_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=False,
             train_flag_dict=feature_dictionary)
        """
