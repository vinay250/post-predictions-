import os
import sys
from source.postpredictions.logger import logging
from source.postpredictions.exception import CustomException
from source.postpredictions.components.data_transformation import DataTransformation
from source.postpredictions.components.model_trainer import ModelTrainer
from source.postpredictions.components.model_evaluation import ModelEvaluation


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebooks\data\dataset1.csv')
            logging.info('Read the dataset as a dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise customexception(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # Perform data transformation
    data_transformation = DataTransformation()
    transformed_data = data_transformation.initialize_data_transformation(train_data, test_data)


    df = pd.read_csv(r'E:\endtoendpost\notebooks\data\dataset1.csv')

    
     # Split the data into features (X) and target variable (y)
    X = df['Facebook Post']
    y = df['Emotion']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Now you can initiate the model training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)
    
    # Initiate model evaluation
    model_evaluation = ModelEvaluation()
    model_evaluation.initiate_model_evaluation(X_train, X_test)

   