import os
import sys
import pickle
import numpy as np
import pandas as pd
from source.postpredictions.logger import logging
from source.postpredictions.exception import customexception

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get classification metrics for the classification task
            acc = accuracy_score(y_test, y_test_pred)
            classification_rep = classification_report(y_test, y_test_pred)
            conf_matrix = confusion_matrix(y_test, y_test_pred)

            report[model_name] = {'accuracy': acc, 'classification_report': classification_rep, 'confusion_matrix': conf_matrix}

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise customexception(e, sys)