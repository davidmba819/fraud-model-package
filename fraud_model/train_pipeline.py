# import necessary files and libraries
import os
import pathlib
import fraud_model
from fraud_model.data_manager import create_train_test_data, save_pipeline
from fraud_model.preprocessing import preprocess_dataset
from fraud_model.pipeline import create_pipeline
from fraud_model.evaluate import evaluate_model
from fraud_model.tracking import MLflowTracker
from fraud_model import config

# create training pipeline
def train_pipeline():
    
    print("Creating train/test datasets...")
    # Load the training data
    train_df, test_df = create_train_test_data()
    
    print('splitting features and target variable...')
    # split the features and target variable
    X_train = train_df.drop(columns=[config.TARGET_FEATURE])
    y_train = train_df[config.TARGET_FEATURE]
    
    # create an instance of MLflowTracker
    print("Setting up MLflow tracking...")
    mlflow_tracker = MLflowTracker(tracking_uri=config.MLFLOW_TRACKING_URI, experiment_name=config.EXPERIMENT_NAME)
    mlflow_tracker.set_experiment()
    mlflow_tracker.start_run()
    
    print("Creating pipeline...")

    # Create the pipeline
    pipeline = create_pipeline()
    
    print("Training model...")

    # train the model
    pipeline.fit(X_train, y_train)
   
    print("Logging parameters to MLflow...") 
    # log parameters to MLflow
    mlflow_tracker.log_parameters(config.XGBOOST_PARAMS)
    
    print("Saving trained pipeline...")
    # save pipeline to disk
    save_pipeline(pipeline)
    
    print("Evaluating model on test data...")
    # evaluate the model
    metrics = evaluate_model(pipeline, test_df)
    
    print("Logging metrics to MLflow...")
    # log metrics to MLflow
    mlflow_tracker.log_metrics({
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"]
    })
    
    print("Logging model to MLflow...")
    # log the trained model to MLflow
    mlflow_tracker.log_model(pipeline)
    
    print("Ending MLflow run...")
    # end the MLflow run
    mlflow_tracker.end_run()
    
    print("Training complete.")
    return pipeline


if __name__ == "__main__":
    train_pipeline()
