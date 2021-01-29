import re
import unicodedata
from string import digits
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk

digits_list = digits


class CleanText:
    def __init__(self):
        french_stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords = [self.remove_accent(sw) for sw in french_stopwords]

        self.stemmer = nltk.stem.SnowballStemmer('english')

    @staticmethod
    def remove_html_code(txt):
        txt = BeautifulSoup(txt, "html.parser", from_encoding='utf-8').get_text()
        return txt

    @staticmethod
    def convert_text_to_lower_case(txt):
        return txt.lower()

    @staticmethod
    def remove_accent(txt):
        return unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode("utf-8")

    @staticmethod
    def remove_non_letters(txt):
        return re.sub('[^a-z_]', ' ', txt)

    def remove_stopwords(self, txt):
        return [w for w in txt.split() if (w not in self.stopwords)]

    def get_stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]


def apply_all_transformation(txt):
    cleaner = CleanText()
    cleaned_txt = cleaner.remove_html_code(txt)
    cleaned_txt = cleaner.convert_text_to_lower_case(cleaned_txt)
    cleaned_txt = cleaner.remove_accent(cleaned_txt)
    cleaned_txt = cleaner.remove_non_letters(cleaned_txt)
    cleaned_txt = cleaner.remove_stopwords(cleaned_txt)
    cleaned_txt = cleaner.get_stem(cleaned_txt)
    return cleaned_txt


def clean_df_column(dataset, column, cleaned_column):
    print("Data Cleaning....")
    dirty_column = dataset[str(column)]
    clean = [" ".join(apply_all_transformation(x)) for x in tqdm(dirty_column)]
    dataset[str(cleaned_column)] = clean
