## This my end to end facebook post prediction project
![Alt text](https://raw.githubusercontent.com/vinay250/post-predictions-/main/photos/Screenshot%20(268).png)

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



 # Facebook Post Sentiment Analysis App

This Flask application demonstrates how to build a web app that analyzes the sentiment of Facebook posts using a pre-trained machine learning model. The app allows users to input a Facebook post, and then displays the predicted sentiment of the post.

## Prerequisites

To run this app, you will need the following:

* Python 3.8 or later
* Flask
* The `source` directory from the provided code

## Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment and activate it.
4. Install the required Python packages by running `pip install -r requirements.txt`.

## Usage

1. Start the Flask app by running `python app.py`.
2. Open your web browser and navigate to `http://localhost:5001`.
3. Enter a Facebook post in the text box and click the "Analyze" button.
4. The predicted sentiment of the post will be displayed on the result page.

## Code Overview

The app consists of the following files:

* `app.py`: The main Flask app file.
* `source/postpredictions/pipelines/prediction_pipeline.py`: The class that encapsulates the machine learning model and prediction logic.
* `source/postpredictions/exception.py`: The custom exception class used to handle errors.
* `source/postpredictions/data/custom_data.py`: The class that represents a custom data object for the prediction pipeline.

### `app.py`

The `app.py` file is the entry point of the app. It sets up the Flask app, configures logging, loads the pre-trained model, and defines the routes for the app.

The `index()` route displays the home page of the app. The `form()` route displays the form where users can input a Facebook post. The `analyze()` route handles the POST request from the form and performs the sentiment analysis.

### `prediction_pipeline.py`

The `prediction_pipeline.py` file defines the `PredictPipeline` class, which encapsulates the machine learning model and prediction logic. The `load_model()` method loads the pre-trained model from a specified path. The `predict()` method takes a `CustomData` object as input and returns the predicted sentiment of the Facebook post.

#overview-domian
## https://dadbxa2u3t.us-east-1.awsapprunner.com/



