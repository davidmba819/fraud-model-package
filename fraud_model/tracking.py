# import libraries
import os
import pathlib

from sklearn import metrics
import fraud_model
from fraud_model import config
import mlflow
import mlflow.sklearn

# create mlflow class
class MLflowTracker:

    """Handles MLflow experiment tracking."""

    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        
    def set_experiment(self):
        """Set the active MLflow experiment."""
        mlflow.set_experiment(self.experiment_name)
        
    def start_run(self):
        """Start an MLflow run."""
        mlflow.start_run()
        
    
        
    def log_parameters(self, parameters):
        """Log model hyperparameters."""
        mlflow.log_params(parameters)
        
    def log_metrics(self, metrics):
        """Log evaluation metrics."""
        mlflow.log_metrics(metrics)
        
    def log_model(self, model):
        """
        Log the trained model as an MLflow artifact.
        """
        mlflow.sklearn.log_model (sk_model=model, artifact_path="fraud_model", serialization_format="pickle")
        
    
        
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()