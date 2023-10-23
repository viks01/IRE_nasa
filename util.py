import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import glob

import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

############################################################ Preprocessing ############################################################
def remove_characters(text: str) -> str:
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_text_files(path: str, N=0, shuffle=False) -> list:
    text_files = glob.glob(f"{path}/*.txt")
    if shuffle:
        random.shuffle(text_files)
    if N == 0:
        N = len(text_files)
    text_files = text_files[:N]
    return text_files

def get_text(text_files: list) -> list:
    text = []
    for text_file in text_files:
        with open(text_file, 'r', errors='ignore') as f:
            content = remove_characters(f.read())
            content = content.lower()
            text.append(content)
    return text

def tokenization(text):
    if type(text) == list:
        return [word_tokenize(t) for t in text]
    elif type(text) == str:
        return word_tokenize(text)
    return None

############################################################ Stemming ############################################################
def stemmer(tokenized_text: list):
    ps = PorterStemmer()
    stemmed_text = []
    for doc in tokenized_text:
        stemmed_text.append([ps.stem(token) for token in doc])

    stemmed_dict = {}
    for doc in stemmed_text:
        for token in doc:
            if token in stemmed_dict:
                stemmed_dict[token] += 1
            else:
                stemmed_dict[token] = 1
    
    return stemmed_dict, stemmed_text

def get_top_stems(stemmed_dict: dict, n: int) -> list:
    sorted_items = sorted(stemmed_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]

############################################################ WordCloud and graphing ############################################################
def plot_wordcloud(items: list):
    word_freq_dict = {word: freq for word, freq in items}
    font_path = "./US101.TTF"
    wordcloud = WordCloud(width=800, height=800, font_path=font_path).generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='none')
    plt.axis('off')
    plt.show()

def plot_tf_dist(items: list):
    word_freq_dict = {word: freq for word, freq in items}
    words = list(word_freq_dict.keys())
    freq = list(word_freq_dict.values())

    fig, ax = plt.subplots(figsize =(20, 20))
    ax.barh(words, freq)

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 1)

    ax.invert_yaxis()

    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round((i.get_width()), 2)), fontsize = 20, fontweight ='bold', color ='grey')
    ax.set_title('Corpus word frequency')

    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.rcParams.update({'font.size': 20})
    plt.show()

############################################################ tf and tf-idf ############################################################
def get_terms_per_doc(tokenized_text: list):
    terms_per_doc = [set(doc) for doc in tokenized_text]
    return terms_per_doc

def get_terms(tokenized_text: list):
    terms = set()
    for doc in tokenized_text:
        for token in doc:
            terms.add(token)
    return list(terms)

# Term Frequency
def get_tf_dict(tokenized_text: list, text_file_names: list, stemming=False):
    tf = {}
    if stemming:
        ps = PorterStemmer()
        for i, doc in enumerate(tokenized_text):
            freq_dict = {}
            for token in doc:
                root = ps.stem(token)
                if root in freq_dict:
                    freq_dict[root] += 1
                else:
                    freq_dict[root] = 1
            file = text_file_names[i]
            tf[file] = freq_dict
    else:
        for i, doc in enumerate(tokenized_text):
            freq_dict = {}
            for token in doc:
                if token in freq_dict:
                    freq_dict[token] += 1
                else:
                    freq_dict[token] = 1
            file = text_file_names[i]
            tf[file] = freq_dict
    return tf

def get_tf_matrix(tf_dict: dict):    
    tf_matrix = pd.DataFrame.from_dict(tf_dict)
    tf_matrix = tf_matrix.fillna(0)
    tf_matrix = tf_matrix / tf_matrix.max()
    return tf_matrix

# Inverse Document Frequency
def get_idf_dict(tokenized_text: list, text_file_names: list, stemming=False):
    if stemming:
        _, tokenized_text = stemmer(tokenized_text)
    terms_per_doc = get_terms_per_doc(tokenized_text)
    terms = get_terms(tokenized_text)
    idf = {}
    N = len(text_file_names)
    for term in terms:
        count = 0
        for doc in terms_per_doc:
            if term in doc:
                count += 1
        idf[term] = np.log(N / count)
    return idf

# Term Frequency - Inverse Document Frequency
def get_tf_idf_matrix(tf_matrix: pd.DataFrame, idf_dict: dict):
    tfidf_matrix = tf_matrix.copy()
    for term in tfidf_matrix.index:
        tfidf_matrix.loc[term] = tfidf_matrix.loc[term] * idf_dict[term]
    return tfidf_matrix

# Stem Analysis
def get_top_stems_per_doc(df: pd.DataFrame, n: int):
    docs = df.columns.values.tolist()
    terms = df.index.values.tolist()
    top_stems = []
    for i in range(len(docs)):
        doc_values = zip(terms, df.iloc[:, i])
        doc_values = sorted(doc_values, key = lambda x : x[1], reverse=True)[:n]
        series = pd.Series([x[1] for x in doc_values], index=[x[0] for x in doc_values])
        top_stems.append(series)
        
    top_stems_by_doc = pd.DataFrame(top_stems)
    top_stems_by_doc = top_stems_by_doc.T
    top_stems_by_doc = top_stems_by_doc.fillna(0)
    top_stems_by_doc.columns = docs
    return top_stems, top_stems_by_doc

def get_avg_stem(df: pd.DataFrame, columnName: str, n=0):
    avg_stem = df.mean(axis=1).to_frame()
    avg_stem.columns = [columnName]
    avg_stem = avg_stem.sort_values(by=[columnName], ascending=False)
    if n > 0:
        avg_stem = avg_stem[:n]
    return avg_stem

# Keyword Comparison
def keyword_comparison(directory: str):
    key_files = glob.glob(f"{directory}/*.key")
    text = get_text(key_files)
    tokenized_text = tokenization(text)
    _, stemmed_text = stemmer(tokenized_text)
    key_files = [Path(file).stem + '.key' for file in key_files]
    stemmed_keyword_dict = {doc: keywords for doc, keywords in zip(key_files, stemmed_text)}
    return stemmed_keyword_dict

def display_keyword_comparison(stemmed_keyword_dict: dict, top_stems: list, top_stems_by_doc: pd.DataFrame, title: str, idx=-1, doc=None):
    if doc is None:
        doc = top_stems_by_doc.columns[idx]
    if idx < 0:
        idx = top_stems_by_doc.columns.tolist().index(doc)
    print(title)
    print(f"Document: {doc}\n")
    print("Top stems in document")
    print(top_stems[idx])
    print("\n\nKeywords")
    doc = Path(doc).stem + '.key'
    print(f"{stemmed_keyword_dict[doc]}\n\n")

# Stem analysis
def display_stem_comparison(top_stems: list, top_stems_by_doc: pd.DataFrame, title: str, idxs=[], docs=[]):
    if docs == []:
        docs = [top_stems_by_doc.columns[idx] for idx in idxs]
    if idxs == []:
        idxs = [top_stems_by_doc.columns.tolist().index(doc) for doc in docs]
    print(title)
    for idx, doc in zip(idxs, docs): 
        print(f"Document: {doc}\n")
        print("Top stems in document")
        print(top_stems[idx])
        print("\n")
    print("\n")

def plot_stem_frequency(top_stems: list, n: int):
    data = {}
    for doc in top_stems:
        stems = doc.index.values.tolist()
        for stem in stems:
            if stem not in data:
                data[stem] = 0
            data[stem] += 1

    keys = list(data.keys())
    values = list(data.values())
    sorted_value_index = np.argsort(values)[::-1][:n]
    data = {keys[i]: values[i] for i in sorted_value_index}
    data = dict(stems=data.keys(), frequency=data.values())
    
    df = pd.DataFrame.from_dict(data)
    plt.figure(figsize=(20, 10))
    sns.barplot(x="stems", y="frequency", data=df)
    plt.xticks(rotation=90)
    plt.show()

############################################################ Model and Queries ############################################################
# Boolean and vector models based on top p stems
def complete_vocabulary(df: pd.DataFrame, top_stems_by_doc: pd.DataFrame):
    top_stems_doc = top_stems_by_doc.copy()
    vocab = df.index.values.tolist()
    current_vocab = set(top_stems_doc.index.values.tolist())
    for term in vocab:
        if term not in current_vocab:
            top_stems_doc.loc[term] = 0.0
    return top_stems_doc

def boolean_model(df: pd.DataFrame, top_stems_by_doc: pd.DataFrame):
    top_stems_doc = complete_vocabulary(df, top_stems_by_doc)
    docs = top_stems_doc.columns.values.tolist()
    terms = top_stems_doc.index.values.tolist()
    boolean_matrix = pd.DataFrame(0, index=terms, columns=docs)
    for doc in docs:
        for term in terms:
            if top_stems_doc.loc[term, doc] > 0:
                boolean_matrix.loc[term, doc] = 1
    return boolean_matrix

def vector_model(df: pd.DataFrame, top_stems_by_doc: pd.DataFrame):
    vector_matrix = complete_vocabulary(df, top_stems_by_doc)
    return vector_matrix

# Query vectorization
def get_query(directory: str, N=1):
    text_files = get_text_files(directory, N=N, shuffle=True)
    text = get_text(text_files)
    text_files = [Path(file).stem + '.txt' for file in text_files]
    query = ""
    for t in text:
        start = random.randint(0, len(t) // 2)
        end = random.randint(start, len(t))
        query += t[start:end]
    query = tokenization(query)
    return query, text_files

def get_query_vector(query: list, df: pd.DataFrame):
    query_vector = pd.Series(0, index=df.index.values.tolist())
    for term in query:
        if term in query_vector.index:
            query_vector.loc[term] += 1
    query_vector = query_vector / query_vector.max()
    return query_vector

# Retrieval and ranking
def retrieval(query_vector: pd.Series, df: pd.DataFrame):
    docs = df.columns.values.tolist()
    terms = df.index.values.tolist()
    retrieval_list = []
    for doc in docs:
        doc_vector = df[doc]
        retrieval_list.append((doc, query_vector.dot(doc_vector)))
    retrieval_list = sorted(retrieval_list, key=lambda x: x[1], reverse=True)
    return retrieval_list

def display_result(query: list, docs: list, retrieval_list: list):
    print(f"Query\n{query}\n")
    print(f"Documents from which query is generated\n{docs}\n")
    print(f"Ranked List\n{retrieval_list}\n")
    for i, (doc, score) in enumerate(retrieval_list):
        if doc in docs:
            print(f"Document {doc} with score {score} is at rank {i + 1}")

def test_model(query: list, docs: list, model: pd.DataFrame, display=True):
    query_vector = get_query_vector(query, model)
    retrieval_list = retrieval(query_vector, model)
    if display:
        display_result(query, docs, retrieval_list)
        return retrieval_list
    result = {}
    for i, (doc, score) in enumerate(retrieval_list):
        if doc in docs:
            result[doc] = i + 1
    return result

############################################################ Stop Words ############################################################
def get_stopwords(path: str):
    with open(path, 'r') as f:
        stopwords = f.read().splitlines()
    return stopwords

def remove_stopwords(path: str, tokenized_text: list):
    stopwords = get_stopwords(path)
    tokenized_text = [[token for token in doc if token not in stopwords] for doc in tokenized_text]
    return tokenized_text

############################################################ Model comparison graph ############################################################
def model_comparison(res1: dict, res2: dict):
    for query in res1:
        data = res1[query]
        df = pd.DataFrame(columns=["model", "rank", "model_type"])
        short_names = {"tf_boolean": "tf_b", "tf_vector": "tf_v", "tfidf_boolean": "tfidf_b", "tfidf_vector": "tfidf_v"}
        for model in data:
            if model == "query" or model == "docs":
                continue
            for doc, rank in data[model].items():
                df.loc[len(df.index)] = [short_names[model], rank, "old"]
        data = res2[query]
        for model in data:
            if model == "query" or model == "docs":
                continue
            for doc, rank in data[model].items():
                df.loc[len(df.index)] = [short_names[model], rank, "new"]
        sns.set(style="ticks")
        sns.catplot(x = "model", y = "rank", hue = "model_type", data = df, kind="box")
        plt.grid()
        plt.show()
