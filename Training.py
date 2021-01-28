import pickle

import tez
import torch
import time
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import numpy as np
from Model.Bert import BERTBaseUncased, BERTDataset


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

        tb_logger = tez.callbacks.TensorBoardLogger(log_dir=".logs/")
        es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path="model_Bert.bin")

        start_time = time.time()

        model.fit(
            processes_data['train_dataset'],
            valid_dataset=processes_data['valid_dataset'],
            train_bs=32,
            device=device,
            epochs=3,
            callbacks=[tb_logger, es],
            fp16=True,
        )
        model.save("model_Bert.bin")
        pred = model.predict(processes_data['test_dataset'])
        results = [torch.argmax(torch.from_numpy(elem), axis=1).numpy() for elem in pred]
        predict_test = []
        for elem in results:
            for e in elem:
                predict_test.append(e)

        score = f1_score(processes_data['y_test'], predict_test, average='macro')
        print(f"Temps Entrainement : {time.time() - start_time} secondes, F1_score = {np.round(score,2)}")
        return model

    def train_SVC(self, processed_data):

        start_time = time.time()
        svc = LinearSVC(max_iter= 500)
        svc.fit(processed_data['X_train'], processed_data['y_train'])
        y_test_pred = svc.predict(processed_data['X_test'])
        score = f1_score(processed_data['y_test'], y_test_pred, average='macro')
        pickle.dump(svc, open("Model/SVC/SVC.pickle", "wb"))
        print(f"Temps Entrainement : {time.time() - start_time} secondes, F1_score = {np.round(score,2)}")
        return svc
