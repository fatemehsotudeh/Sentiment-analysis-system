import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np



def vectorize_bow(train_data, val_data, test_data):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_val = vectorizer.transform(val_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_val, X_test


def vectorize_tfidf(train_data, val_data, test_data):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_val = vectorizer.transform(val_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_val, X_test


def vectorize_word2vec_data(data):
    tokenized_data = data.apply(word_tokenize)
    word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
    word2vec = tokenized_data.apply(
        lambda x: np.mean([word2vec_model.wv[word] for word in x if word in word2vec_model.wv], axis=0) if x and any(
            word in word2vec_model.wv for word in x) else np.zeros(word2vec_model.vector_size))
    return pd.DataFrame(word2vec.tolist())


def load_glove_model(file_path):
    model = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            model[word] = vector
    return model


def vectorize_glove(train_data, val_data, test_data, glove_model, vector_size=50):
    def vectorize_text_data(text_data):
        vectorized_data = []
        for text in text_data:
            tokens = word_tokenize(text.lower())
            word_vectors = [glove_model.get(word, np.zeros(vector_size)) for word in tokens]
            text_vector = np.mean(word_vectors, axis=0)
            vectorized_data.append(text_vector)
        return np.array(vectorized_data)

    X_train = vectorize_text_data(train_data)
    X_val = vectorize_text_data(val_data)
    X_test = vectorize_text_data(test_data)

    return X_train, X_val, X_test
