import torch

from Cleaning_text.CleanText import clean_df_column
import numpy as np
import pandas as pd
import pickle
from Model.Bert.Bert import BERTDataset, BERTBaseUncased


class Prediction:

    def prediction_Bert(self, dataset):

        SEED = 1
        cuda = torch.cuda.is_available()
        torch.manual_seed(SEED)
        device = "cuda" if cuda else "cpu"

        train_label = pd.read_csv("app/dataset/train_label.csv")
        print(train_label.Category.nunique())
        model = BERTBaseUncased(num_train_steps=int(len(dataset) / 32 * 10),
                                num_classes=train_label.Category.nunique())
        model.load("model_Bert.bin", device=device)
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
        bert_pred.to_csv("Bert_Predictions.csv", index=False)
        return bert_pred

    def prediction_SVC(self, dataset):
        model = pickle.load(open("app/Model/SVC/SVC.pickle", "rb"))
        Count_Vect = pickle.load(open("app/Model/SVC/Count_Vect.pickle", "rb"))
        clean_df_column(dataset, 'description', 'description_cleaned')
        dataset_vect = Count_Vect.transform(dataset["description_cleaned"].values)
        dataset["Category"] = model.predict(dataset_vect)
        svc_pred = dataset[["Id", "Category"]]
        #svc_pred.to_csv("Svc_Predictions.csv", index=False)
        return svc_pred