import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Assuming you have defined `categorical_features` and `numerical_features` somewhere in your code.

# Identify categorical and numerical features
categorical_features = ['Facebook Post']
numerical_features = ['User ID']

# Create transformers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder())
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Create a pipeline with the preprocessor and a classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',  MultinomialNB())
])

# Assuming you have loaded your data somewhere in your code
# For example:
# X = data['Facebook Post']
# y = data['Emotion']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of your training and testing data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
