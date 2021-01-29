import gdown
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


class Procesing4BLCG:

    def __init__(self, path):
        self.path = path
        self.embedding_path = 'app/Model/BLCG/glove.840B.300d.txt'

    def load_train(self, path):

        print("_________load train data____________")
        train_df = pd.read_json(self.path + "/train.json")
        train_label = pd.read_csv(self.path + "/train_label.csv")
        train_df = train_df.set_index('Id', drop=False)
        train_df.index.name = None

        return train_df, train_label

    def load_test_data(self, path):
        print("_________load test data____________")
        test_df = pd.read_json(self.path + "/test.json")
        test_df = test_df.set_index('Id', drop=False)
        test_df.index.name = None

        return test_df

    def tokenizer_data(self, X_train, X_valid, X_test, max_features, max_len):

        print("_________Start tokenize data____________")

        tk = Tokenizer(num_words=max_features, lower=True)
        raw_text_train = X_train["description"].str.lower()
        raw_text_valid = X_valid["description"].str.lower()
        raw_text_test = X_test["description"].str.lower()

        tk.fit_on_texts(raw_text_train)

        X_train.loc[:, "description_seq"] = tk.texts_to_sequences(raw_text_train)
        X_valid.loc[:, "description_seq"] = tk.texts_to_sequences(raw_text_valid)
        X_test.loc[:, "description_seq"] = tk.texts_to_sequences(raw_text_test)

        X_train = pad_sequences(X_train.description_seq, maxlen=max_len)
        X_valid = pad_sequences(X_valid.description_seq, maxlen=max_len)
        X_test = pad_sequences(X_test.description_seq, maxlen=max_len)

        return X_train, X_valid, X_test

    def tokenizer_test(self, X_train, X_test, max_features, max_len):

        print("_________Start tokenize data____________")

        tk = Tokenizer(num_words=max_features, lower=True)
        raw_text_train = X_train["description"].str.lower()
        raw_text_test = X_test["description"].str.lower()

        tk.fit_on_texts(raw_text_train)

        X_train.loc[:, "description_seq"] = tk.texts_to_sequences(raw_text_train)
        X_test.loc[:, "description_seq"] = tk.texts_to_sequences(raw_text_test)

        X_train = pad_sequences(X_train.description_seq, maxlen=max_len)
        X_test = pad_sequences(X_test.description_seq, maxlen=max_len)

        return X_train, X_test

    def tokenizer(self, X_train, max_features, max_len):
        tk = Tokenizer(num_words=max_features, lower=True)
        raw_text_train = X_train["description"].str.lower()

        tk.fit_on_texts(raw_text_train)

        return tk

    def get_coefs(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def vectorization(self, X_train, embed_size, max_features, max_len):

        print("______________________________Glove: Start vectorisation______________________________")

        try:
            embedding_index = dict(
                self.get_coefs(*o.strip().split(" ")) for o in open(self.embedding_path, encoding="utf8"))
        except:
            print("Glove not found downloading it from Drive (5 GB) .........")
            url = "https://drive.google.com/uc?id=15R5mkGUo3OJKvSXj-EYkucwiY1aXsNau"
            output = self.embedding_path
            gdown.download(url, output, quiet=False)

        finally:
            embedding_index = dict(
                self.get_coefs(*o.strip().split(" ")) for o in open(self.embedding_path, encoding="utf8"))
            tk = self.tokenizer(X_train, max_features, max_len)
            word_index = tk.word_index
            nb_words = min(max_features, len(word_index))
            embedding_matrix = np.zeros((nb_words, embed_size))
            for word, i in word_index.items():
                if i >= max_features: continue
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            print("______________________________Glove: End vectorisation______________________________")

            return embedding_matrix
