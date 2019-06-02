"""
vs_search.py
author: Daniel Zhang
"""


from vs_index import word_normalization, read_data
from nltk.corpus import stopwords
import nltk
import shelve
from collections import Counter
from collections import defaultdict
import  math
from operator import itemgetter
import heapq
import string

def get_title_text(doc_id):
    """This method receives the document id and return text and title of the document as a string """
    data=read_data("doc-data.json")

    text = data.get(doc_id).get("Text")

    title = data.get(doc_id).get("Title")

    return title[0] + text



def make_query_tfidf(query):
    """This method take query string and process it including removing stop words etc., then calculate the tf-idf and return it as a dictionary."""

    inverted_idx=shelve.open("shelve_file.db", flag="r")
    # get the doc-id list for the main query(title and text)

    document_length = shelve.open("doc_length_shelve_file.db", flag= "r")

    total_document = len(document_length)
    ps = nltk.stem.PorterStemmer()

    query_tokens_list = []
    for i in nltk.word_tokenize(query):
        if i.lower() not in stopwords.words('english') and i not in string.punctuation and ps.stem(i.lower()) in inverted_idx.keys():
            query_tokens_list.append(i)


    token_frequency_dict = dict(Counter(query_tokens_list))

    for token, value in token_frequency_dict.items():
        term = ps.stem(token.lower())
        tf = math.log(value+1)
        idf = math.log(total_document/len(inverted_idx.get(term)))
        token_frequency_dict[token] = tf*idf

    document_length.close()
    inverted_idx.close()

    return token_frequency_dict


def get_cosine_similarity(query_tfidf):
    """This method take the query’s tf-idf vector, get posting list associated with each term and calculate the cosine similarity score for each document which contains at least one word in the query."""
    cosine_matching_result = defaultdict(lambda: [0, []])
    inverted_idx = shelve.open("shelve_file.db", flag="r")
    document_length = shelve.open("doc_length_shelve_file.db", flag= "r")
    ps = nltk.stem.PorterStemmer()


    for token, query_tfidf in sorted(query_tfidf.items(), key= lambda x: x[0]):
        term = ps.stem(token.lower())

        posting_list = inverted_idx.get(term)
        for doc_id, doc_tfidf in posting_list:

            doc_length = document_length.get(doc_id)

            cosine_matching_result[doc_id][0]+= doc_tfidf*query_tfidf/doc_length

            cosine_matching_result[doc_id][1].append(token)


    return cosine_matching_result

def get_k_largest_docid(cosine_matching_result,k):
    """this method receive the dictionary of each documents cosines similariry score and use Heapq in python to get the k documents with largest cosine similarity score."""
    result  = heapq.nlargest(k, cosine_matching_result.items(), key=lambda x: x[1][0])

    return result

def get_movie_data(doc_id):
    """
    Return data fields including title, text, director, starring and country information for a movie.
    This method uses the doc_id as the key to access the shelf entry for the movie doc_data.
    """

    s=read_data("doc-data.json")

    movie=s.get(doc_id)
    title = movie.get("Title")[0]
    director =  ", ".join(movie.get("Director"))
    starring = ", ".join(movie.get("Starring"))
    country = ", ".join(movie.get("Country"))
    location = ", ".join(movie.get("Location"))
    text = movie.get("Text")




    movie_object = {"title": title,
                    "director": director,
                    "starring": starring,
                    "country": country,
                    "location": location,
                    "text": text
                    }


    return movie_object


def get_movie_snippet(doc_id):
    """
    This method will return a title and a snippet of the movie text for the results page.
    """
    s=read_data("doc-data.json")

    movie = s.get(doc_id)

    title = movie.get("Title")[0]

    sent_list = nltk.sent_tokenize(str(movie.get("Text")))
    if len(sent_list)<=5:
        text = movie.get("Text")
    else:
        text = " ".join(sent_list[0:5])

    return (doc_id, title, text)



