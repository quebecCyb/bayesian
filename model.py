import pandas as pd
import nltk
import ssl
from nltk.stem import *
from nltk.stem.porter import *
import numpy as np
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
stop_words = set(stopwords.words("english"))


VOCAB_LEN = 2500

class Model:
    _instance = None
    classes_prob = []
    word_classes_prob = []
    word_prob = {}
    class_names = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
            cls._instance.initialize()  # You can add initialization code here
        return cls._instance

    def initialize(self):
        # Initialization code goes here
        pass

    def set_train_data(self, class_names, classes_prob, word_classes_prob, word_prob):
        self.class_names = class_names
        self.classes_prob = classes_prob
        self.word_classes_prob = word_classes_prob
        self.word_prob = word_prob
    
    def forward(self, email):
        classifier = [ 0 for _ in self.classes_prob ]
        for word in clean_message(email):
            if word not in self.word_prob.index:
                continue
            for i in range(len(self.classes_prob)):
                classifier[i] += np.log(self.word_classes_prob[i][word]) + np.log(1 - self.classes_prob[i]) - np.log(self.word_prob[word])
        print(classifier)
        print(self.class_names)
        return self.class_names[classifier.index(max(classifier))]


def clean_message(message, ps=PorterStemmer()):
    message = BeautifulSoup(f'{message}', 'html.parser').get_text()
    return list(filter(lambda x: ps.stem(x) not in stop_words and x.isalpha(), word_tokenize(message.lower())))


def train():

    df = pd.read_csv('data.csv', encoding='latin1',usecols=['v1', 'v2'])

    df = pd.DataFrame(data={ "text": [ i for i in df.v2 ], 'label': [ i for i in df.v1 ] })

    tokenized_data = df.text.apply(clean_message)
    vocab_list = [i for sublist in tokenized_data for i in sublist]

    vocab_df = pd.Series(vocab_list).value_counts()
    vocab_ids = list(range(VOCAB_LEN))

    vocab_df = pd.DataFrame({'VOCAB_WORD': vocab_df.sort_values(ascending=False).head(2500).index.values}, index=vocab_ids)

    def create_full_matrix():
        data = {'DOC_ID': [i for i in df.index]  , 'LABEL': [i for i in df.label], }
        for i in vocab_df.VOCAB_WORD:
            data[i] = [tokenized_data.values[j].count(i) for j in df.index]

        _df = pd.DataFrame(data)
        return _df

    full_matrix = create_full_matrix()

    unique_classes = list(full_matrix.groupby('LABEL').groups.keys())
    classes_prob = []
    classes_word_prob = []

    for i in range(len(unique_classes)):
        classes_prob.append(full_matrix[full_matrix.LABEL == unique_classes[i]].LABEL.size / full_matrix.LABEL.size)

        train_local = full_matrix[full_matrix.LABEL == unique_classes[i]]
        summed_local_tokens = train_local.sum(axis=0)
        summed_local_tokens = summed_local_tokens.drop('LABEL')
        summed_local_tokens = summed_local_tokens.drop('DOC_ID')

        classes_word_prob.append(summed_local_tokens / summed_local_tokens.sum())

    summed_tokens = full_matrix.sum(axis=0)
    summed_tokens = summed_tokens.drop('LABEL')
    summed_tokens = summed_tokens.drop('DOC_ID')

    prob_word = summed_tokens / summed_tokens.sum()

    m = Model()
    m.set_train_data(unique_classes, classes_prob, classes_word_prob, prob_word)
