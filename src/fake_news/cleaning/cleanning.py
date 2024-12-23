
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class CleannedText:
    def __init__(self):
        self.stop_words = list(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.texts = []

    def remove_characters_special(self, text):
        return re.sub(r'[^a-zA-Z\s]', '', text)
    
    def to_lower(self, text):
        return text.lower()
    
    def tokenize(self, text):
        return nltk.word_tokenize(text)
    
    def remove_stop_words(self, texts):
        for y in texts:
            if y not in self.stop_words:
                self.texts.append(self.lemmatizer.lemmatize(y))

    def join_text(self):
        return ' '.join(self.texts)
    
    def clean_text(self, text):
        text = self.remove_characters_special(text)
        text = self.to_lower(text)
        texts = self.tokenize(text)
        self.remove_stop_words(texts)
        text = self.join_text()
        return text