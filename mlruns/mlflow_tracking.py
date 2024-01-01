# mlflow/mlflow_tracking.py
import mlflow

def train_model():
    # Your ML training code here
    with mlflow.start_run():
        # Log metrics, parameters, artifacts, etc.
        mlflow.log_param("param_name", param_value)
        mlflow.log_metric("metric_name", metric_value)
        mlflow.log_artifact("file_path_or_directory")