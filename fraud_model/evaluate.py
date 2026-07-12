# import libraries
import os
import pathlib
from mlflow import metrics
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import tracker
from fraud_model.data_manager import load_pipeline, create_train_test_data, load_test_data
from fraud_model.preprocessing import preprocess_dataset
from fraud_model.predict import predict
from fraud_model import config
from fraud_model.tracking import MLflowTracker



# calculate metrics for the model
def cal_metrics(y_true, y_pred):
    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    return metrics_dict

# evaluate the model
def evaluate_model(pipeline, test_df):
    
    # split the features and target variable
    X_test = test_df.drop(columns=[config.TARGET_FEATURE])
    y_test = test_df[config.TARGET_FEATURE]
    
    # make predictions
    print("Making predictions...")
    y_pred = predict(pipeline, X_test)
    
    # evaluate the model
    print("Evaluating model...")
    metrics  = cal_metrics(y_test, y_pred)
    
    print("\nModel Evaluation")
    print("-" * 30)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
 
    return metrics
    

    