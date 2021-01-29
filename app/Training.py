import pickle

import tez
import torch
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from Model.Bert.Bert import BERTBaseUncased

from Model.BLCG.Model import Model4BLCG
from Model.BLCG.processing import  Procesing4BLCG
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score


class Training:

    def train_Bert(self, processes_data):

        SEED = 1
        cuda = torch.cuda.is_available()
        torch.manual_seed(SEED)
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            torch.cuda.manual_seed(SEED)

        model = BERTBaseUncased(num_train_steps=processes_data['n_train_steps'],
                                num_classes=processes_data['train_label'].Category.nunique())

        tb_logger = tez.callbacks.TensorBoardLogger(log_dir="../.logs/")
        es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path="app/Model/Bert/model_Bert.bin")

        start_time = time.time()

        model.fit(
            processes_data['train_dataset'],
            valid_dataset=processes_data['valid_dataset'],
            train_bs=16,
            device=device,
            epochs=3,
            callbacks=[tb_logger, es],
            fp16=True,
        )
        model.save("app/Model/Bert/model_Bert.bin")

        pred = model.predict(processes_data['test_dataset'])
        results = [torch.argmax(torch.from_numpy(elem), axis=1).numpy() for elem in pred]
        predict_test = []
        for elem in results:
            for e in elem:
                predict_test.append(e)

        score = f1_score(processes_data['y_test'], predict_test, average='macro')
        print(f"Temps Entrainement : {time.time() - start_time} secondes, F1_score = {np.round(score, 2)}")
        return score

    def train_SVC(self, processed_data):

        start_time = time.time()
        svc = LinearSVC(max_iter=500)
        svc.fit(processed_data['X_train'], processed_data['y_train'])
        y_test_pred = svc.predict(processed_data['X_test'])
        score = f1_score(processed_data['y_test'], y_test_pred, average='macro')
        pickle.dump(svc, open("app/Model/SVC/SVC.pickle", "wb"))
        print(f"Temps Entrainement : {time.time() - start_time} secondes, F1_score = {np.round(score, 2)}")
        return score


class TrainingBi:

    def __init__(self):
        self.path = 'app/dataset'
        self.model_path = 'app/Model/BLCG/Bi_GRU.hdf5'
        self.embed_size = 300
        self.max_features = 130000
        self.max_len = 220

    def training_Bi_Gru_Lstm_Cnn_Glove(self):
        start_time = time.time()

        print(" __________________________________________________________________________________________")
        print("____________________ Bi_Gru_Lstm_Cnn_Glove: Starting training ______________________________")
        print(" __________________________________________________________________________________________")

        model4blcg = Model4BLCG(self.path)
        proces = Procesing4BLCG(self.path)

        # load data
        train_df, train_label = proces.load_train(self.path)

        print(" Train Shape : " + str(train_df.shape[0]), str(train_df.shape[1]))

        # split data
        X_train_test, X_valid, Y_train_test, Y_valid = train_test_split(train_df, train_label
                                                                        , test_size=0.2, random_state=13)

        X_train, X_test, Y_train, Y_test = train_test_split(X_train_test, Y_train_test,
                                                            test_size=0.2, random_state=13)

        # cleaning and vectorize data
        X_train_new, X_valid_new, X_test_new = proces.tokenizer_data(X_train, X_valid, X_test, self.max_features,
                                                                     self.max_len)

        Y_train = Y_train.Category.values
        Y_valid = Y_valid.Category.values
        Y_test = Y_test.Category.values

        model = model4blcg.build_models(X_train, X_train_new, Y_train, X_valid_new, Y_valid)

        print(" Training Completed with computational time : {}".format(time.time() - start_time))

        print(" _______________ Start evaluating the model________________")

        y_pred = model.predict(X_test_new)
        y_pred = np.argmax(y_pred, axis=1)

        # Print accuracy, f1, precision, and recall scores
        print(" test accuracy :", accuracy_score(Y_test, y_pred))
        print(" test precision :", precision_score(Y_test, y_pred, average="macro"))
        print(" test recall :", recall_score(Y_test, y_pred, average="macro"))
        print(" test f1_score :", f1_score(Y_test, y_pred, average="macro"))

        print(" ________________________________________________________________________________________")
        print("_____________________ Bi_Gru_Lstm_Cnn_Glove: Ending training ______________________________")
        print(" ________________________________________________________________________________________")

        return f1_score(Y_test, y_pred, average="macro")
