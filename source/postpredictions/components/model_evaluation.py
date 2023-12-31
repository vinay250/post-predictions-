import os
from urllib.parse import urlparse
import mlflow
from mlflow import log_artifacts
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd  # Import pandas if not already imported

class ModelEvaluation:
    def __init__(self):
        pass
    
    def log_info(self, message):
        mlflow.log_text("info", message)

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            # Assuming 'Emotion' is the target column
            target_column = 'Emotion'

            # Ensure test_array is a DataFrame
            if not isinstance(test_array, pd.DataFrame):
                raise ValueError("The 'test_array' is not a DataFrame.")

            # Check if the target column is present in the DataFrame
            if target_column not in test_array.columns:
                raise ValueError(f"The target column '{target_column}' is not present in the DataFrame.")

            # Split features (X_test) and target variable (y_test) from the test_array
            X_test = test_array.drop(columns=[target_column])
            y_test = test_array[target_column]

            # Load the trained model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            # Set the MLflow registry URI
            mlflow.set_registry_uri("https://dagshub.com/vinayakavirat008/post-predictions-.mlflow")
            
            # Get the type of the tracking URI
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            self.log_info(f"Tracking URL Type: {tracking_url_type_store}")

            # Start a new MLflow run
            with mlflow.start_run():
                print(mlflow.get_tracking_uri())

                self.log_info("Making predictions using the test set")
                predicted_qualities = model.predict(X_test)

                self.log_info("Evaluating the model and logging classification metrics")
                accuracy = accuracy_score(y_test, predicted_qualities)
                classification_rep = classification_report(y_test, predicted_qualities)
                confusion_mat = confusion_matrix(y_test, predicted_qualities)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)

                # If using a remote registry, register the model in the Model Registry
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    # If using a local file store, log the model without registering
                    mlflow.sklearn.log_model(model, "model")

                # Log classification metrics
                mlflow.log_text("classification_report", classification_rep)
                mlflow.log_artifact(confusion_mat, "confusion_matrix.txt")
        
        except Exception as e:
            self.log_info(f"Exception occurred: {str(e)}")
            # Handle the exception as needed

# Replace 'your_module_containing_load_object' with the actual module where load_object is defined

# Example usage:
#model_evaluation = ModelEvaluation()
#model_evaluation.initiate_model_evaluation(X_train, X_test)
