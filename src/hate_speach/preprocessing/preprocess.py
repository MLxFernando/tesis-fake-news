import pandas as pd
import sys
import os
from dotenv import load_dotenv
import joblib


load_dotenv()

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(os.getenv("PYTHONPATH"))
print("PYTHONPATH:", os.getenv("PYTHONPATH"))
from src.hate_speach.utils.constants import HATE_SPEACH_PATH, HATE_SPEACH_PROCESSED_PATH, HATE_SPEACH_VECTORIZER_PATH

from src.hate_speach.extraction_features.tfid_vectorizer import TfidfFeatureExtractor
from src.hate_speach.extraction_features.sv_extraction import SentimentFeatureExtractor
from src.hate_speach.extraction_features.addition_features import AdditionalFeatureExtractor
from src.hate_speach.extraction_features.features_combine import FeatureCombiner
from src.hate_speach.cleanning.cleanning import CleanningData

class PreprocessData:
    def __init__(self):
        self.tfidf_extractor = TfidfFeatureExtractor()
        self.sentiment_extractor = SentimentFeatureExtractor()
        self.additional_extractor = AdditionalFeatureExtractor()

    def cleanning_and_preprocessing_data_for_trainner(self):
        self.data_general = pd.read_csv(HATE_SPEACH_PATH)
        self.data_general['text length'] = self.data_general['tweet'].apply(len)
        self.cleaning_data = CleanningData(self.data_general['tweet'])
        self.data_cleanning = self.cleaning_data.cleaning_data()

        # Extraer características con cada método
        self.tfidf_features = self.tfidf_extractor.extract_features(self.data_cleanning)
        self.tfidf_extractor.save_vectorizer(HATE_SPEACH_VECTORIZER_PATH)
        self.sentiment_features = self.sentiment_extractor.extract_features(self.data_general['tweet'])
        self.additional_features = self.additional_extractor.extract_features(self.data_cleanning)

    def combine_feature_for_trainner(self):
        # Limpieza de datos
        self.feature_combiner = FeatureCombiner()
        self.combined_features = self.feature_combiner.combine_features(self.tfidf_features, self.sentiment_features, self.additional_features)  
        self.combined_features_df = pd.DataFrame(self.combined_features)
        self.combined_features_df['class'] = self.data_general['class']
        self.combined_features_df.to_csv(HATE_SPEACH_PROCESSED_PATH, index=False)
        
    def cleanning_and_preprocessing_data_for_predict(self, data):
        print("Texto Original: "+data)
        self.data = pd.Series([data])
        self.cleaning_data = CleanningData(self.data)
        self.data_cleanning = self.cleaning_data.cleaning_data()
        print("Texto Limpiado: "+self.data_cleanning)
        vectorizer = joblib.load(HATE_SPEACH_VECTORIZER_PATH)
        self.tfidf_features = self.tfidf_extractor.transform( self.data_cleanning, vectorizer)
        self.sentiment_features = self.sentiment_extractor.extract_features(self.data)
        self.additional_features = self.additional_extractor.extract_features(self.data_cleanning)

    def combine_feature_for_predict(self):
        self.feature_combiner = FeatureCombiner()
        self.combined_features = self.feature_combiner.combine_features(self.tfidf_features, self.sentiment_features, self.additional_features)  
        self.combined_features_df = pd.DataFrame(self.combined_features)
        return self.combined_features_df


if __name__ == "__main__":
    preprocess = PreprocessData()
    preprocess.cleanning_and_preprocessing_data_for_trainner()
    preprocess.combine_feature_for_trainner()
    print(preprocess.combined_features_df.head())