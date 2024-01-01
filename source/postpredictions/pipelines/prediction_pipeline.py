import os
import sys
import pandas as pd
from source.postpredictions.exception import customexception
from source.postpredictions.logger import logging
from source.postpredictions.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            scaled_data = preprocessor.transform(features)
            
            pred = model.predict(scaled_data)
            
            return pred
        
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self, facebook_post, emotion, user_id):
        self.facebook_post = facebook_post
        self.emotion = emotion
        self.user_id = user_id

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Facebook Post': [self.facebook_post],
                'Emotion': [self.emotion],
                'User ID': [self.user_id]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occurred in creating dataframe')
            raise customexception(e, sys)
