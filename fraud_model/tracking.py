# import libraries
import os
import pathlib

from sklearn import metrics
import fraud_model
from fraud_model import config
import mlflow

# create mlflow class
class MLflowTracker:

    """Handles MLflow experiment tracking."""

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        
    def set_experiment(self):
        """Set the active MLflow experiment."""
        mlflow.set_experiment(self.experiment_name)
        
    def start_run(self):
        """Start an MLflow run."""
        mlflow.start_run()
        
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        
    def log_parameters(self, parameters):
        """Log model hyperparameters."""
        mlflow.log_params(parameters)
        
    def log_metrics(self, metrics):
        """Log evaluation metrics."""
        mlflow.log_metrics(metrics)