import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from source.postpredictions.logger import logging
from source.postpredictions.exception import customexception

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from source.postpredictions.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Facebook Post']
            numerical_cols = ['User ID']
            
            logging.info('Pipeline Initiated')
            
            # Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical Pipeline
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            # Text Pipeline
            text_pipeline = Pipeline(steps=[
                ('vectorizer', CountVectorizer())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols),
                ('text_pipeline', text_pipeline, 'Facebook Post')
            ])

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred in the get_data_transformation")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Facebook Post']
            numerical_cols = ['User ID']
            
            # Pipeline for numerical features
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Pipeline for categorical features
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            
            # Preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )

            target_column_name = 'Emotion'
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # Use train_df and test_df directly instead of input_feature_train_df and input_feature_test_df
            input_feature_train_arr = preprocessor.fit_transform(train_df)
            input_feature_test_arr = preprocessor.transform(test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            logging.info("Preprocessing pickle file saved")
            
            return train_arr, test_arr
            
        except Exception as e:
            logging.info("Exception occurred in the initialize_data_transformation")
            raise customexception(e, sys)
