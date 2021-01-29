import time

import torch

from Cleaning_text.CleanText import clean_df_column
import numpy as np
import pandas as pd
import pickle
from Model.Bert.Bert import BERTDataset, BERTBaseUncased
from sklearn.model_selection import train_test_split
from keras.models import load_model
from Model.BLCG.processing import Procesing4BLCG
import gdown



class Prediction:

    def prediction_Bert(self, path="app/dataset/test.json"):

        dataset = pd.read_json(path)
        SEED = 1
        cuda = torch.cuda.is_available()
        torch.manual_seed(SEED)
        device = "cuda" if cuda else "cpu"

        model = BERTBaseUncased(num_train_steps=int(len(dataset) / 32 * 10),
                                num_classes=28)
        try:
            model.load("app/Model/Bert/model_Bert.bin", device=device)
        except:
            print("Model not found downloading it from Drive.........")
            url = "https://drive.google.com/uc?id=1GabihQ76KtH8_i5GV09uzCWmu1phlCK0"
            output = 'app/Model/Bert/model_Bert.bin'
            gdown.download(url, output, quiet=False)

        finally:
            model.load("app/Model/Bert/model_Bert.bin", device=device)
            clean_df_column(dataset, 'description', 'description_cleaned')

            tar = np.full((len(dataset.description_cleaned.values),), 0)
            test_dataset = BERTDataset(text=dataset.description_cleaned.values, target=tar)
            pred = model.predict(test_dataset)
            results_pred = [torch.argmax(torch.from_numpy(elem), axis=1).numpy() for elem in pred]
            predict_csv = []

            for elem in results_pred:
                for e in elem:
                    predict_csv.append(e)

            dataset["Category"] = predict_csv
            bert_pred = dataset[["Id", "Category"]]
            bert_pred.to_csv("app/Predictions/Bert_Predictions.csv", index=False)
            return bert_pred

    def prediction_SVC(self, path="app/dataset/test.json"):
        dataset = pd.read_json(path)
        try:
            model = pickle.load(open("app/Model/SVC/SVC.pickle", "rb"))
        except:
            print("Model not found downloading it from Drive.........")
            url = "https://drive.google.com/uc?id=1pyMl0mzqOlNfQLDchCKXk8GFpALkqON1"
            output = 'app/Model/SVC/SVC.pickle'
            gdown.download(url, output, quiet=False)
        finally:
            model = pickle.load(open("app/Model/SVC/SVC.pickle", "rb"))
            try:
                TF_IDF = pickle.load(open("app/Model/SVC/Count_Vect.pickle", "rb"))
            except:
                print("Vectorizer not found downloading it from Drive.........")
                url = "https://drive.google.com/uc?id=174qD1VjjenpPlf-LaEjYnUBT9PGAOgZz"
                output = "app/Model/SVC/Count_Vect.pickle"
                gdown.download(url, output, quiet=False)
            finally:
                TF_IDF = pickle.load(open("app/Model/SVC/Count_Vect.pickle", "rb"))
                clean_df_column(dataset, 'description', 'description_cleaned')
                dataset_vect = TF_IDF.transform(dataset["description_cleaned"].values)
                dataset["Category"] = model.predict(dataset_vect)
                svc_pred = dataset[["Id", "Category"]]
                svc_pred.to_csv("app/Predictions/Svc_Predictions.csv", index=False)
                return svc_pred


class PredictionBi:

    def __init__(self):
        self.path = 'app/dataset'
        self.model_path = 'app/Model/BLCG/Bi_GRU.hdf5'
        self.embed_size = 300
        self.max_features = 70000
        self.max_len = 220

    def predict_Bi_Gru_Lstm_Cnn_Glove(self):
        print(" __________________________________________________________________________")
        print(" _______________ Bi_Gru_Lstm_Cnn_Glove: Starting prediction ________________")
        print(" __________________________________________________________________________")

        start_time = time.time()

        proces = Procesing4BLCG(self.path)

        # load data
        test_df = proces.load_test_data(self.path)
        train_df, _ = proces.load_train(self.path)

        print(" Test dataset Shape : " + str(test_df.shape[0]), str(test_df.shape[1]))

        X_train_test, X_valid = train_test_split(train_df, test_size=0.2, random_state=13)

        X_train, X_test = train_test_split(X_train_test, test_size=0.2, random_state=13)

        _, X_test_new = proces.tokenizer_test(X_train, test_df, self.max_features, self.max_len)

        try:
            model = load_model(self.model_path)
        except:
            print("Model not found downloading it from Drive.........")
            url = "https://drive.google.com/uc?id=1uWPLF2MDPvq4YOOc57xhnNSP2aeCBVvw"
            output = self.model_path
            gdown.download(url, output, quiet=False)
        finally:
            model = load_model(self.model_path)
            y_pred_test = model.predict(X_test_new)
            predictions = np.argmax(y_pred_test, axis=1)

            test_df["Category"] = predictions
            prediction_file = test_df[["Id", "Category"]]

            prediction_file.to_csv("app/Predictions/bi_gru_lstm_cnn_glove.csv", index=False)

            print(" Training Completed with computational time : {}".format(time.time() - start_time))

            print(" ________________________________________________________________________")
            print(" _______________ Bi_Gru_Lstm_Cnn_Glove: Ending prediction ________________")
            print(" ________________________________________________________________________")

            return prediction_file
