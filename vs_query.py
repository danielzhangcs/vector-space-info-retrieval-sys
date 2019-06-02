"""
vs_query.py
Dependencies: python 3.x, flask

To start the application:
   >python boolean_query.py
To terminate the application, use control-c
To use the application within a browser, use the url:
   http://127.0.0.1:5000/

"""

import nltk
import shelve
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from vs_search import *
from vs_index import *

# Create an instance of the flask application within the appropriate namespace (__name__).
# By default, the application will be listening for requests on port 5000 and assuming the base
# directory for the resource is the directory where this module resides.
app = Flask(__name__)


# Welcome page
# Python "decorators" are used by flask to associate url routes to functions.
# A route is the path from the base directory (as it would appear in the url)
# This decorator ties the top level url "localhost:5000" to the query function, which
# renders the query_page.html template.
@app.route("/")
def query():
    """For top level route ("/"), simply present a query page."""
    return render_template('query_page.html')


# This takes the form data produced by submitting a query page request and returns a page displaying
# results (SERP).
@app.route("/results", methods=['POST'])
def results():
    """Generate a result set for a query and present the 10 results starting with <page_num>."""
    shelf = shelve.open("shelve_file.db", flag="r")

    ps = nltk.stem.PorterStemmer()
    page_num = int(request.form['page_num'])

    if "more_like_this" in request.form:
        doc_id = request.form["more_like_this"]
        query = get_title_text(doc_id)
    else:
        query = request.form['query']


      # Get the raw user query

    original_query_terms = [word for word in set(nltk.word_tokenize(query)) if word not in string.punctuation]
    # Keep track of any stop words removed from the query to display later.
    # Stop words should be stored in persistent storage when building your index,
    # and loaded into your search engine application when the application is started.
    skipped = [e for e in original_query_terms if ps.stem(e.lower()) in stopwords.words('english')]

    stopping_removed = [e for e in original_query_terms if ps.stem(e.lower()) not in skipped]


    unknown_terms = [e for e in stopping_removed if ps.stem(e.lower()) not in list(shelf.keys())]


    # If your search found any query terms that are not in the index, add them to unknown_terms and
    # render the error_page.

    # At this point, your query should contain normalized terms that are not stopwords or unknown.
    query_tfidf = make_query_tfidf(query)
    cosine_matching_result = get_cosine_similarity(query_tfidf)
    movie_ids_and_score= get_k_largest_docid(cosine_matching_result, page_num*10)
      # Get a list of movie doc_ids that satisfy the query.
    # render the results page
    num_hits = len(cosine_matching_result)
    movie_ids = [i[0] for i in movie_ids_and_score[page_num*10-10: page_num*10]]# Limit of 10 results per page
    cosine_score = [i[1][0] for i in movie_ids_and_score[page_num*10-10: page_num*10]]

    matched_term_list = [i[1][1] for i in movie_ids_and_score[page_num*10-10: page_num*10]]
    # movie_results = list(map(dummy_movie_snippet, movie_ids))  # Get movie snippets: title, abstract, etc.
    # Using list comprehension:
    movie_results = [get_movie_snippet(e) for e in movie_ids]
    return render_template('results_page.html', text_query=query, movie_results=movie_results,
                               srpn=page_num,
                               len=len(movie_results), skipped_words=skipped, unknown_terms=unknown_terms,
                               total_hits=num_hits, cosine =cosine_score, matched = matched_term_list)


# Process requests for movie_data pages
# This decorator uses a parameter in the url to indicate the doc_id of the film to be displayed
@app.route('/movie_data/<film_id>')
def movie_data(film_id):
    """Given the doc_id for a film, present the title and text (optionally structured fields as well)
    for the movie."""
    data = get_movie_data(film_id)  # Get all of the info for a single movie
    return render_template('doc_data_page.html', data=data)


# If this module is called in the main namespace, invoke app.run().
# This starts the local web service that will be listening for requests on port 5000.
if __name__ == "__main__":
    app.run(debug=True)
    # # While you are debugging, set app.debug to True, so that the server app will reload
    # # the code whenever you make a change.  Set parameter to false (default) when you are
    # # done debugging.


