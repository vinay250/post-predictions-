#this my end to end facebook post prediction project

#first initilize the git
step by step overview of project 



or create a new repository on the command line
echo "# post-predictions-" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin 
git push -u origin main

…or push an existing repository from the command line
git remote add origin https://github.com/
git push -u origin main


bash your_file_name.sh


python setup.py install

# another way can mention  -e . in you requiremnt file and you can run

pip install -r requiremnets.txt

# Project Overview

This project aims to predict emotions from social media posts. The following components are part of the project:

## Data Ingestion

The data ingestion process involves:

1. Reading the dataset as a dataframe.
2. Performing a train-test split on the dataset.
3. Saving the raw data, train data, and test data.
4. Applying preprocessing on training and testing datasets using a preprocessor object.
5. Saving the preprocessing object for future use.

### Dataset Overview

The dataset consists of the following columns:

- Facebook Post: The text of the social media post.
- Emotion: The emotion label associated with the post.
- User ID: The user ID associated with the post.

### Example Train Data

| Facebook Post                      | Emotion  | User ID |
|------------------------------------|----------|---------|
| Enjoying a peaceful evening.       | Peaceful | 678901  |
| Excited for the weekend!            | Excited  | 456789  |
| Had a tough day at school. :(      | Sad      | 789012  |

### Example Test Data

| Facebook Post                       | Emotion  | User ID |
|-------------------------------------|----------|---------|
| Starting a new hobby. Excited!"     | Excited  | 789012  |
| Feeling happy today! :)             | Happy    | 123456  |
| Worried about the future."          | Worried  | 901234  |

## Model Training

The model training process involves:

1. Initializing the data transformation and model trainer components.
2. Performing data transformations, including imputation and encoding.
3. Splitting the data into features (X) and the target variable (y).
4. Training a Naive Bayes classifier.
5. Evaluating the model's accuracy, precision, recall, F1-score, and confusion matrix.

### Model Report

The Naive Bayes model achieved an accuracy of 66.67%. The classification report and confusion matrix provide insights into the model's performance on different emotions.


