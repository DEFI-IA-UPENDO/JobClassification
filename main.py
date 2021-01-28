import torch

from Prediction import Prediction
from Processing import Processing
from Training import Training
import pandas as pd


SEED = 1
cuda = torch.cuda.is_available()
torch.manual_seed(SEED)
device = torch.device("cuda" if cuda else "cpu")
print(device)

if __name__ == '__main__':
    process = Processing(path='dataset')
    training = Training()
    prediction = Prediction()

    ##########################   BERT    ######################################

    processed_bert = process.process_Bert()
    model_bert = training.train_Bert(processes_data=processed_bert)
    """
    test = pd.read_json("dataset/test.json")
    predictions_test = prediction.prediction_Bert(dataset=test)
     """
    ##########################   SVC    ########################################
    """
    processed_svc = process.process_SVC()
    model_svc = training.train_SVC(processed_data=processed_svc)
    test = pd.read_json("dataset/test.json")
    predictions_test = prediction.prediction_SVC(dataset=test,model= model_svc)
    """



