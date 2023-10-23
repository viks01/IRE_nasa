from util import *

from createTermDocumentMatrix import createTermDocumentMatrix
from queryBooleanRepresentation import queryBooleanRepresentation
from queryVectorRepresentation import queryVectorRepresentation
from compQueryBoolean import compQueryBoolean
from compQueryVector import compQueryVector

directory = "./nasa"
stop_words_path = "./english.stop"
tf_boolean_matrix, tfidf_boolean_matrix, _ = createTermDocumentMatrix(directory, stop_words_path, boolean=True, stemming=True)
tf_vector_matrix, tfidf_vector_matrix, idf_dict = createTermDocumentMatrix(directory, stop_words_path, boolean=False, stemming=True)

# For query, N = number of documents to use to generate the query
query, docs = get_query(directory, N=50)
vocabulary = tf_boolean_matrix.index.values.tolist()
query_boolean = queryBooleanRepresentation(query, vocabulary)
query_vector = queryVectorRepresentation(query, idf_dict)

# Number of ranked documents retrieved
N = 10
print("Term Frequency Boolean Model With Boolean Query")
compQueryBoolean(query_boolean, docs, tf_boolean_matrix, N=N)
print("\n\n\nTerm Frequency Vector Model With Vector Query")
compQueryVector(query_vector, docs, tf_vector_matrix, N=N)
print("\n\n\nTF-IDF Boolean Model With Boolean Query")
compQueryBoolean(query_boolean, docs, tfidf_boolean_matrix, N=N)
print("\n\n\nTF-IDF Vector Model With Vector Query")
compQueryVector(query_vector, docs, tfidf_vector_matrix, N=N)
