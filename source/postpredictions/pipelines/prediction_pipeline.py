import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from source.postpredictions.exception import CustomException
from source.postpredictions.logger import logging
from source.postpredictions.utils.utils import load_object, save_object
from dataclasses import dataclass

@dataclass
class PredictPipline:
    Pipline_model_file = os.path.join('artifacts','Pipline.pkl')
    
class PredictPipeline:
    def __init__(self):
        self.model_loaded = False  # Assume the model is not loaded initially
        self.Pipline = None

    def load_model(self, model_path):
        try:
            # Load the pipeline from files
            Pipline_path = os.path.join(model_path, "Pipline.pkl")
            self.Pipline = load_object(pipeline_path)

            self.model_loaded = True  # Set the flag to indicate that the model is loaded
            logging.info('Model loaded successfully')

        except Exception as e:
            logging.error('Error loading model')
            raise CustomException(e, sys)

    def predict(self, custom_data):
        try:
            if not self.model_loaded:
                raise CustomException("Model not loaded. Call load_model() first.")

            # Assuming 'text' is the input text for prediction
            text = custom_data.facebook_post  # Assuming the text is stored in 'facebook_post' attribute

            # Make prediction
            try:
                pred = self.Pipline.predict([text])
            except Exception as e:
                # Handle the case where the prediction itself causes an error
                raise CustomException(f"Error during prediction: {e}")

            return pred

        except CustomException as ce:
            # Catch custom exceptions and print detailed information
            print(f"Custom Exception Details: {ce}")
            raise ce
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, facebook_post, emotion, user_id):
        self.facebook_post = facebook_post
        self.emotion = emotion
        self.user_id = user_id

    def get_features(self):
        # Return a dictionary containing the values of the features
        return {'facebook_post': self.facebook_post, 'user_id': self.user_id}
