import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from Prediction import Prediction, PredictionBi
from Processing import Processing
from Training import Training, TrainingBi
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from Voting.voting import make_final_prediction
import warnings

warnings.filterwarnings("ignore")

SEED = 1
cuda = torch.cuda.is_available()
torch.manual_seed(SEED)
device = torch.device("cuda" if cuda else "cpu")

process = Processing(path='app/dataset')
training = Training()
prediction = Prediction()

training_Bi = TrainingBi()
prediction_Bi = PredictionBi()




def predict():

    path = "app/dataset/test.json"
    dataset = pd.read_json(path)
    print("----------------SVC----------------")
    svc_pred = prediction.prediction_SVC()
    print("---------------SVC done--------------")

    print("----------------BLCG----------------")
    lstm_gru_cnn_glove_pred = prediction_Bi.predict_Bi_Gru_Lstm_Cnn_Glove()
    print("---------------BLCG done--------------")

    print("---------------BERT----------------")
    bert_pred = prediction.prediction_Bert()
    print("---------------BERT done--------------")

    results = make_final_prediction(bert_pred["Category"], svc_pred["Category"], lstm_gru_cnn_glove_pred["Category"])
    dataset["Category"] = results
    predictions_csv = dataset[["Id", "Category"]]
    predictions_csv.to_csv("app/Predictions/final_prediction.csv", index=False)
    return results


if __name__ == '__main__':

    print("JobClassification project (Valdom) by : \n"
          "GHOMSI KONGA Serge \n"
          "KEITA Alfousseyni \n"
          "RIDA Moumni \n"
          "SANOU Désiré \n"
          "WAFFA PAGOU Brondon \n")

    print(" Please select one of the following features : \n"
          "1 - Make a prediction \n"
          "2 - Train a model \n")
    S = int(input("Choix : "))
    if S is 1:
        predict()
    elif S is 2:

        print("Which model do you want to train : \n"
              "1 - Support Vector Machine \n"
              "2 - Bert \n"
              "3 - Bi_GRU \n")
        S = int(input("Choix : "))

        if S is 1:
            print("Data Preprocessing ....................")
            processed_svc = process.process_SVC()
            print("Training Model ........................")
            score_svc = training.train_SVC(processed_data=processed_svc)
            print(f"Training DONE !!")
        elif S is 2:
            print("Data Preprocessing ....................")
            processed_bert = process.process_Bert()
            print("Training Model ........................")
            score_bert = training.train_Bert(processes_data=processed_bert)
            print(f"Training DONE !!")
        elif S is 3:
            score_Bi = training_Bi.training_Bi_Gru_Lstm_Cnn_Glove()
            print(f"Training DONE !!2")



