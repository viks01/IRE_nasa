from util import *

# Similarity measure is the same as retrieval function in util.py
def compQueryVector(query: pd.Series, docs: list, model: pd.DataFrame, display=True, N=0):
    retrieval_list = retrieval(query, model)
    if N > 0:
        retrieval_list = retrieval_list[:N]
    if display:
        display_result(query, docs, retrieval_list)
        return retrieval_list
    result = {}
    for i, (doc, score) in enumerate(retrieval_list):
        if doc in docs:
            result[doc] = i + 1
    return result