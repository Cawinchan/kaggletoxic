import csv
import os
import pickle
import time

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer


def main(data_file, w2v_model, testing, expt_name="test"):
    if testing:
        print("running tests")
        number_of_epochs = 1
    else:
        print("running eval")
        number_of_epochs = 50

    # load data
    print("loading data")
    full_data, text_data = load_data(data_file)

    # process data
    print("processing data")
    tokenized_sentences = tokenize_tweets(text_data)
    removed_indexes, vectorized_sentences_np = vectorise_tweets(w2v_model, tokenized_sentences)
    safe_remove_indexes_from_list(removed_indexes,
                                  full_data)  # because some text return nothing, must remove ground truth too
    x_test, x_train, y_test, y_train = split_train_test(full_data, vectorized_sentences_np)

    # build neural network model
    model = build_keras_model()
    print("training network")
    history = model.fit(x_train, y_train,
                        batch_size=128, epochs=number_of_epochs,
                        validation_data=(x_test, y_test))

    # try some values
    x_predict = model.predict(x_train[::1000])
    for index, val in enumerate(x_predict):
        print("predicted is {}, truth is {},".format(x_predict[index][0], y_train[index]))
    save_model_details_and_training_history(expt_name, history, model)


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime("%d_%m_%y_%H:%M") + "_" + expt_name
    os.makedirs('keras_models/{}'.format(folder), exist_ok=True)
    model.save('keras_models/{}/keras_model.h5'.format(folder))
    with open('keras_models/{}/trainHistoryDict'.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def load_w2v_model_from_path(model_path, binary_input=False):
    """
    :param model_path: path to w2v model
    :type model_path: string
    :param binary_input: True : binary input, False : text input
    :type binary_input: boolean
    :return: loaded w2v model
    :rtype: KeyedVectors object
    """
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=binary_input)
    return w2v_model

def build_keras_model():
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    data_dim = 300
    timesteps = 50
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def split_train_test(full_data, vectorized_sentences_np):
    one_tenth_size = len(vectorized_sentences_np)//10
    x_train = vectorized_sentences_np[one_tenth_size:]
    x_test = vectorized_sentences_np[:one_tenth_size]
    y_train = full_data['toxic'][one_tenth_size:]
    y_test = full_data['toxic'][:one_tenth_size]
    return x_test, x_train, y_test, y_train


def vectorise_tweets(model, tokenized_sentences):
    # vectorise sentences
    removed_indexes = []
    vectorized_sentences = []
    for i in range(len(tokenized_sentences)):
        tokenized_sentence = tokenized_sentences[i]
        if len(tokenized_sentence) > 50:
            tokenized_sentence = tokenized_sentence[:50]
        vector_rep_of_sentence = []
        for word in tokenized_sentence:
            if word in model.vocab:
                vector_rep_of_sentence.append(model.wv[word])
        if not vector_rep_of_sentence:
            removed_indexes.append(i)
        else:
            array = np.array(vector_rep_of_sentence)
            zeroes = np.zeros((50 - len(vector_rep_of_sentence), 300))
            vector_rep_of_sentence = np.concatenate((array, zeroes), axis=0)
            vectorized_sentences.append(vector_rep_of_sentence)
    vectorized_sentences_np = np.array(vectorized_sentences)
    return removed_indexes, vectorized_sentences_np


def tokenize_tweets(text_data):
    tknzr = TweetTokenizer()
    max_length = 0
    # tokenize sentences
    tokenized_sentences = []
    for sentence in text_data:
        tokenized_sentences.append(tknzr.tokenize(sentence))
        max_length = max(max_length, len(sentence))
    return tokenized_sentences


def load_data(data_file):
    full_data_set = []
    with open(data_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            full_data_set.append(line)
    # load data into native lists
    print(header)
    id_data = [i for i in map(lambda x: x[0], full_data_set)]
    text_data = [i for i in map(lambda x: x[1], full_data_set)]
    toxic_data = [i for i in map(lambda x: x[2], full_data_set)]
    severe_toxic_data = [i for i in map(lambda x: x[3], full_data_set)]
    obscene_data = [i for i in map(lambda x: x[4], full_data_set)]
    threat_data = [i for i in map(lambda x: x[5], full_data_set)]
    insult_data = [i for i in map(lambda x: x[6], full_data_set)]
    identity_hate_data = [i for i in map(lambda x: x[6], full_data_set)]
    full_data = {'id': id_data, 'toxic': toxic_data, 'severe_toxic': severe_toxic_data, 'obscene': obscene_data,
                 'threat': threat_data, 'insult': insult_data, 'identity_hate': identity_hate_data}
    return full_data, text_data


def safe_remove_indexes_from_list(list_of_indexes, full_data_set):
    list_of_indexes.sort(reverse=True)  # always remove the largest indexes first or you will get an index error
    for key in full_data_set:  # for each sequence
        sequence = full_data_set[key]
        for index in list_of_indexes:  # iterate through index
            sequence.pop(index)
        full_data_set[key] = sequence


if __name__ == "__main__":
    EXPT_NAME = "TEST"
    SAMPLE_DATA_FILE = './data/sample.csv'
    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    main(SAMPLE_DATA_FILE, model, testing=True, expt_name=EXPT_NAME)

    print("done with tests, loading true model")
    EXPT_NAME = "REAL"
    DATA_FILE = './data/train.csv'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    model = load_w2v_model_from_path(W2V_MODEL)
    main(DATA_FILE, model, testing=False, expt_name=EXPT_NAME)