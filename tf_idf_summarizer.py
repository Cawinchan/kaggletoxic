import re

import nltk
from nltk.corpus import stopwords

stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np


# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/


def summarize_long_sentences(array_of_strings, max_size=300, max_sentences=10):
    assert isinstance(array_of_strings, np.ndarray)
    assert type(array_of_strings[0]) == str
    print("starting summarization")
    # Load corpus array_of_strings used to train the TF-IDF Transformer
    training_data = []
    for index, document in enumerate(array_of_strings):
        cleaned_document = clean_document(document)
        training_data.append(cleaned_document)
    print("done with cleaning")
    # Merge corpus array_of_strings and new document array_of_strings
    # Fit and Transform the term frequencies into a vector
    count_vect = CountVectorizer(stop_words='english')
    count_vect = count_vect.fit(training_data)
    freq_term_matrix = count_vect.transform(training_data)
    feature_names = count_vect.get_feature_names()
    # Fit and Transform the TfidfTransformer
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
    print("done with fitting")
    return_list = []
    # Get the dense tf-idf matrix for the document
    count = 0
    split_data_and_shorten_if_needed(count, count_vect, feature_names, max_sentences, return_list, tfidf, training_data)
    new_list = []
    for result in return_list:
        assert type(result[1]) == str
        new_list.append(result[1])
    assert len(array_of_strings) == len(new_list)
    print(new_list)
    return new_list


def split_data_and_shorten_if_needed(count, count_vect, feature_names, max_sentences, return_list, tfidf,
                                     training_data):
    for index, document in enumerate(training_data):
        if index % 10000 == 0:
            print("progress : {} out of {}".format(index, len(training_data)))
        tokenized_sentence = document.split()
        if len(tokenized_sentence) > 300:
            count += 1
            tokenized_sentence_by_sentence = nltk.sent_tokenize(document)
            if len(tokenized_sentence_by_sentence) == 1 or len(tokenized_sentence_by_sentence) >= 60:  # treash sentence
                return_list.append((index, " ".join(training_data[index].split()[:300])))
            else:
                return_list.append((index, parallel_tf_idf(count_vect, training_data[index], document,
                                                           feature_names,
                                                           max_sentences, tfidf)))
        else:
            return_list.append((index, training_data[index]))
    return_list.sort(key=lambda x: x[0])


def parallel_tf_idf(count_vect, cleaned_document, document, feature_names, max_sentences, tfidf):
    story_freq_term_matrix = count_vect.transform([document])
    story_tfidf_matrix = tfidf.transform(story_freq_term_matrix)
    story_dense = story_tfidf_matrix.todense()
    doc_matrix = story_dense.tolist()[0]
    # Get Top Ranking Sentences and join them as a summary
    top_sents = rank_sentences(document, doc_matrix, feature_names, top_n=max_sentences)
    summary = "\n".join([nltk.sent_tokenize(cleaned_document)[i] for i in top_sents])
    return summary


def clean_document(document):
    """Cleans document by removing unnecessary punctuation. It also removes
    any extra periods and merges acronyms to prevent the tokenizer from
    splitting a false sentence
    """
    # Remove all characters outside of Alpha Numeric
    # and some punctuation
    document = document.replace('-', '')
    document = document.replace('...', '')
    document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    # Remove Ancronymns M.I.T. -> MIT
    # to help with sentence tokenizing
    document = merge_acronyms(document)

    # Remove extra whitespace
    document = ' '.join(document.split())
    return document


def remove_stop_words(document):
    """Returns document without stop words"""
    document = ' '.join([i for i in document.split() if i not in stop])
    return document


def similarity_score(t, s):
    """Returns a similarity score for a given sentence.
    similarity score = the total number of tokens in a sentence that exits
                        within the title / total words in title
    """
    t = remove_stop_words(t.lower())
    s = remove_stop_words(s.lower())
    t_tokens, s_tokens = t.split(), s.split()
    similar = [w for w in s_tokens if w in t_tokens]
    score = (len(similar) * 0.1) / len(t_tokens)
    return score


def merge_acronyms(s):
    """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.', ''))
    return s


def rank_sentences(doc, doc_matrix, feature_names, top_n=3):
    """Returns top_n sentences. Theses sentences are then used as summary
    of document.
    input
    ------------
    doc : a document as type str
    doc_matrix : a dense tf-idf matrix calculated with Scikits TfidfTransformer
    feature_names : a list of all features, the index is used to look up
                    tf-idf scores in the doc_matrix
    top_n : number of sentences to return
    """
    sents = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    """
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                  for sent in sentences]

    # Calculate Sentence Values
    doc_val = sum(doc_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]
    # Apply Position Weights
    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    print(ranked_sents)
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] * -1)
    selected_sents = ranked_sents[:top_n]
    sentence_indexes = [i[0] for i in selected_sents]
    """
    sentence_indexes = [0, len(sentences) - 1]
    set_sentences = [set(i) for i in sentences]
    index_sentence_set = set()
    for index in sentence_indexes:
        index_sentence_set.update(set_sentences[index])
    for i in range(len(set_sentences)):
        if i in sentence_indexes:
            continue
        sentence = set_sentences[i]
        combined_set = sentence.intersection(index_sentence_set)
        if len(combined_set) < (len(sentence) * 1 // 4):  # at least 1/4 of the words are novel
            sentence_indexes.append(i)
            index_sentence_set.update(sentence)
    return sorted(sentence_indexes)


if __name__ == "__main__":
    from utils import load_data, COMMENT_TEXT_INDEX

    data_file = './data/balanced_train_file.csv'
    df = load_data(data_file)
    list_of_documents = df[COMMENT_TEXT_INDEX].values
    documents = summarize_long_sentences(list_of_documents)
    size = {}
    for index, document in enumerate(documents):
        length = len(document.split())
        if length in size:
            size[length] += 1
        else:
            size[length] = 1
        if length > 600:
            list_of_documents[index] = " ".join(document.split()[:300])
    size_list = [i for i in size.items()]
    size_list.sort(key=lambda x: x[0], reverse=True)
    print(size_list)
    import pickle

    pickle.dump(documents, open("./data/tf_balanced_train.p", "wb"))
