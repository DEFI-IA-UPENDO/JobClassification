import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from Cleaning_text.CleanText import clean_df_column
from Model.Bert import BERTDataset
import pickle


class Processing:

    def __init__(self, path):
        self.path = path
        self.train_df = pd.read_json(self.path + "/train.json")
        self.train_label = pd.read_csv(self.path + "/train_label.csv")
        self.categories_string = pd.read_csv(self.path + "/categories_string.csv")

    def process_Bert(self):
        clean_df_column(self.train_df, 'description', 'description_cleaned')

        X_train, X_test, y_train, y_test = train_test_split(self.train_df.description_cleaned.values,
                                                            self.train_label.Category.values, test_size=0.20,
                                                            random_state=42)

        train_dataset = BERTDataset(text=X_train, target=y_train)
        test_dataset = BERTDataset(text=X_test, target=np.full((len(X_test),), 0))
        valid_dataset = BERTDataset(text=X_test, target=y_test)
        n_train_steps = int(len(X_train) / 32 * 10)

        processes_data = {
            'train_dataset': train_dataset,
            'valid_dataset': valid_dataset,
            'test_dataset': test_dataset,
            'y_test': y_test,
            'n_train_steps': n_train_steps,
            'train_label': self.train_label
        }
        return processes_data

    def process_SVC(self):
        clean_df_column(self.train_df, 'description', 'description_cleaned')
        Count_Vect = CountVectorizer(ngram_range=(1, 2))
        Count_Vect.fit(self.train_df["description_cleaned"].values)

        pickle.dump(Count_Vect, open("Model/SVC/Count_Vect.pickle", "wb"))

        train_df_vect = Count_Vect.transform(self.train_df["description_cleaned"].values)


        X_train, X_test, y_train, y_test = train_test_split(train_df_vect, self.train_label.Category.values,
                                                            test_size=0.2,
                                                            random_state=42)

        processed_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        return processed_data
