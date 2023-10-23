from util import *

def queryVectorRepresentation(query: list, idf_dict: dict):
    query_dict = {}
    for term in query:
        if term not in query_dict:
            query_dict[term] = 1
        else:
            query_dict[term] += 1
            
    query_vector = {}
    for term in idf_dict:
        if term in query_dict:
            query_vector[term] = query_dict[term] * idf_dict[term]
        else:
            query_vector[term] = 0
    return pd.Series(query_vector)