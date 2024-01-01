# prediction_pipeline.py
import sys
import os
import pandas as pd
import numpy as np
import sys  # Add this import statement
from sklearn.base import is_classifier
from sklearn.utils.validation import check_array
from source.postpredictions.exception import customexception
from source.postpredictions.logger import logging
from source.postpredictions.utils.utils import load_object, save_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, custom_data):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            # Get features from CustomData
            features = custom_data.get_features()
            
            # Convert the list to a NumPy array and reshape
            features_array = np.array(features).reshape(-1, 1)
            
            # Perform necessary preprocessing
            scaled_data = preprocessor.transform(features_array)
            
            # Make prediction
            pred = model.predict(scaled_data)
            
            return pred
        
        except Exception as e:
            raise customexception(e, sys)
    
    def save_model(self, save_path):
        try:
            # Save the preprocessor and model to files
            preprocessor_path = os.path.join(save_path, "preprocessor.pkl")
            model_path = os.path.join(save_path, "model.pkl")

            save_object(self.preprocessor, preprocessor_path)
            save_object(self.model, model_path)

            logging.info('Model saved successfully')

        except Exception as e:
            logging.error('Error saving model')
            raise customexception(e, sys)

    @classmethod
    def load_model(cls, load_path):
        try:
            preprocessor_path = os.path.join(load_path, "preprocessor.pkl")
            model_path = os.path.join(load_path, "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            pipeline = cls()
            pipeline.preprocessor = preprocessor
            pipeline.model = model

            logging.info('Model loaded successfully')

            return pipeline

        except Exception as e:
            logging.error('Error loading model')
            raise customexception(e, sys)

# CustomData class in prediction_pipeline.py
class CustomData:
    def __init__(self, facebook_post, emotion, user_id):
        self.facebook_post = facebook_post
        self.emotion = emotion
        self.user_id = user_id

    def get_features(self):
        # Assuming you want to use only the 'facebook_post' as the feature
        return [[self.facebook_post, "", ""]]