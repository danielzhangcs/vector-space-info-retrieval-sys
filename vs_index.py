"""
vs_index.py
author: Daniel Zhang
"""
import json
import nltk
import string
from nltk.corpus import stopwords
from collections import defaultdict
import shelve
from collections import Counter
import math
import timeit




def read_data(file):
    """This method is responsible for reading json file into python and load it as a dictionary"""
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def create_word_frequency_list(data):
    """This method will call methods to do normalization for all the words in a document
    and create word frequency dictionary for each documents"""
    word_frequency_list= []

    for i in range(1, len(data)+1):

        text = data.get(str(i)).get("Text")

        title = data.get(str(i)). get("Title")

        temp = word_normalization(nltk.word_tokenize(title[0])+ nltk.word_tokenize(text))

        word_frequency_dict=dict(Counter(temp))

        word_frequency_list.append(word_frequency_dict)

    return word_frequency_list


def remove_punctuation(word_list):
    """This method is responsible for removing all the punctuations in corpus of each movie"""
    return [word for word in word_list if word not in string.punctuation]


def make_lowercase(word_list):
    """This method is resposible for case fold"""
    return [word.lower() for word in word_list]


def remove_stopwords(word_list):
    """This method is responsible for removing stop words in the corpus of each movie data"""
    return [word for word in word_list if word not in stopwords.words('english')]

def stemming_words(word_list):
    """This method using Porter stemmer to stem each word in corpus of each movie so that terms are made"""
    ps = nltk.stem.PorterStemmer()
    return [ps.stem(word) for word in word_list]


def word_normalization(word_list):
    """This method call all the method responsible for normalizing the data"""

    punctuation_removed=remove_punctuation(word_list)

    case_lowered = make_lowercase(punctuation_removed)

    stopwords_removed = remove_stopwords(case_lowered)

    stemmed_words = stemming_words(stopwords_removed)

    return stemmed_words


def make_inverted_index(word_frequency_list):
    """This method create term posting list for each term and the element in each posting list is a tuple which contains the document id and occurence frequency"""
    frequency_inv_indx = defaultdict(list)


    for idx, word_frequency_dict in enumerate(word_frequency_list,1):

        for keys, value in word_frequency_dict.items():

            if keys != '[]':
                frequency_inv_indx[keys].append((str(idx),value))

    return frequency_inv_indx


def make_tfidf_inverted_idx(frequency_inv_indx, total_doc_length):
    """This method receive the inverted frequency index dictionary and convert frequency in each tuple into tf-idf score. """
    for term, posting_list in frequency_inv_indx.items():
        document_frequency = len(posting_list)
        idf = math.log(total_doc_length / document_frequency, 2)

        for idx, value in enumerate(posting_list):
            tf = math.log(value[1]+1, 2)
            posting_list[idx] = (value[0], tf*idf)


    return frequency_inv_indx


def make_document_length(word_frequency_list,inv_idx):
    """This method use tf-idf vector of each document and calculate the length of each document used for cosine normalization and will be stored as a dictionary."""
    document_length_idx = {}
    total_doc_length = len(word_frequency_list)

    for idx, i in enumerate(word_frequency_list,1):
        length=0

        for key, value in i.items():
            tf = math.log(value+1, 2)
            idf = math.log(total_doc_length / len(inv_idx.get(key)), 2)

            length+= math.pow(tf*idf, 2)
        document_length_idx[str(idx)] = math.sqrt(length)

    return document_length_idx


def create_shelve_file(shelve_file, dict):
    """This method store the inverted index into a shelve file so that user don't have to build index for
    query purpose each time they want to make a query"""
    for key, value in dict.items():
        shelve_file[key] = value





if __name__ == "__main__":

    data=read_data("doc-data.json")

    word_frequency_list = create_word_frequency_list(data)

    frequency_inv_indx=make_inverted_index(word_frequency_list)

    inv_indx = make_tfidf_inverted_idx(frequency_inv_indx, len(word_frequency_list))

    shelve_file = shelve.open("shelve_file.db", flag='c')

    start = timeit.default_timer()

    create_shelve_file(shelve_file, inv_indx)

    shelve_file.close()

    stop = timeit.default_timer()

    print(stop-start)





    document_length_idx = make_document_length(word_frequency_list, inv_indx)

    doc_length_shelve_file = shelve.open("doc_length_shelve_file.db", flag='c')

    create_shelve_file(doc_length_shelve_file, document_length_idx)

    doc_length_shelve_file.close()








