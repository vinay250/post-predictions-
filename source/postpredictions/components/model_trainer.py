import os
import pandas as pd
import sys
import logging
from dataclasses import dataclass
from source.postpredictions.logger import logging
from source.postpredictions.exception import customexception
from source.postpredictions.utils.utils import save_object
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            # Naive Bayes model training
            vectorizer = CountVectorizer()
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)

            classifier = MultinomialNB()
            classifier.fit(X_train_vectorized, y_train)

            # Make predictions on the test set
            y_test_pred = classifier.predict(X_test_vectorized)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_test_pred)
            print(f'Accuracy: {accuracy:.2f}')

            classification_rep = classification_report(y_test, y_test_pred)
            conf_matrix = confusion_matrix(y_test, y_test_pred)

            report = {
                'NaiveBayes': {
                    'accuracy': accuracy,
                    'classification_report': classification_rep,
                    'confusion_matrix': conf_matrix
                }
            }

            print(report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {report}')

            # Save the Naive Bayes model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=classifier
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise customexception(e, sys)

if __name__ == "__main__":
    df = pd.read_csv(r'E:\endtoendpost\notebooks\data\dataset1.csv')

    
     # Split the data into features (X) and target variable (y)
    X = df['Facebook Post']
    y = df['Emotion']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Now you can initiate the model training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)