import torch

from Prediction import Prediction
from Processing import Processing
from Training import Training
import pandas as pd
import tkinter as tk
from tkinter import filedialog

SEED = 1
cuda = torch.cuda.is_available()
torch.manual_seed(SEED)
device = torch.device("cuda" if cuda else "cpu")

process = Processing(path='app/dataset')
training = Training()
prediction = Prediction()


def train_models():
    ### Data Cleaning ###
    print("Data Preprocessing ....................")

    processed_bert = process.process_Bert()
    processed_svc = process.process_SVC()

    ## Model Training ######

    print("Training Models ........................")

    score_svc = training.train_SVC(processed_data=processed_svc)
    score_bert = training.train_Bert(processes_data=processed_bert)

    print("Training DONE !!")


def predict():
    print("Chemin du fichier : ")
    path = filedialog.askopenfilename()
    print(path)
    dataset = pd.read_json(path)

    predict_svc = prediction.prediction_SVC(dataset)
    predict_bert = prediction.prediction_Bert(dataset)


if __name__ == '__main__':

    print("JobClassification project (Valdom) by : \n"
          "GHOMSI KONGA Serge \n"
          "KEITA Alfousseyni \n"
          "RIDA Moumni \n"
          "SANOU Désiré \n"
          "WAFFA PAGOU Brondon \n")

    print("Veuillez selectionner l'une des fonctionnalitées suivantes : \n"
          "1 - Effectuer une prédication (Dataset en format json) \n"
          "2 - Ré-entrainer le modèle \n")
    S = int(input("Choix : "))
    if S is 1:
        predict()
    elif S is 2:
        train_models()
