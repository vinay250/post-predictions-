from source.postpredictions.components.data_ingestion import DataIngestion
from source.postpredictions.components.data_transformation import DataTransformation
from source.postpredictions.components.model_trainer import ModelTrainer
from source.postpredictions.components.model_evaluation import ModelEvaluation

import os
import sys
from source.postpredictions.logger import logging
from source.postpredictions.exception import customexception
import pandas as pd
from sklearn.model_selection import train_test_split  # Add this import

def main():
    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        # Perform data transformation
        data_transformation = DataTransformation()
        transformed_data = data_transformation.initialize_data_transformation(train_data, test_data)

        # Read data from CSV file
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

    except Exception as e:
        logging.error('An exception occurred: {}'.format(str(e)))
        raise customexception(e, sys)

if __name__ == "__main__":
    main()
