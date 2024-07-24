import pandas
import re
import nltk
import pymorphy2

from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, stopwords, regex=re.compile('\w+')):
        self.stopwords = stopwords
        self.regex = regex

    def find_words(self, text):
        return re.findall(self.regex, text)

    def lemmatize(self, words):
        morph = pymorphy2.MorphAnalyzer()
        lemmas = [morph.parse(w)[0].normal_form for w in words if not w in self.stopwords]

        return lemmas

    def preprocess_text(self, text):
        return self.lemmatize(self.find_words(text))

    def preprocess_labels(self, labels):
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)

        y2label = {index: label for index, label in enumerate(le.classes_)}

        return y2label, target
