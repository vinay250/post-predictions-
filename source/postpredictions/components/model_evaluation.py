import os
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from source.postpredictions.utils.utils import load_object
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Retrieve MLflow tracking URI, username, and password from environment variables
mlflow_tracking_uri = os.environ.get("https://dagshub.com/vinayakavirat008/post-predictions-.mlflow")
mlflow_tracking_username = os.environ.get("vinayakavirat008")
mlflow_tracking_password = os.environ.get("8006a8c9e77082e90d96573a13bb278110094dff")

class ModelEvaluation:
    def __init__(self):
        pass
    
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            # Split features (X_test) and target variable (y_test) from the test_array
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Load the trained model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            # Set the MLflow registry URI
            mlflow.set_registry_uri("https://dagshub.com/vinayakavirat008/post-predictions-.mlflow")
            
            # Get the type of the tracking URI
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)

            # Start a new MLflow run
            with mlflow.start_run():

                # Make predictions using the test set
                predicted_qualities = model.predict(X_test)

                # Evaluate the model and log classification metrics
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
            raise e
