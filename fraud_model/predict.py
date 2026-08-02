# import necessary libraries
from fraud_model import config
from fraud_model.data_manager import load_pipeline
import pandas as pd
import numpy as np

 # load the trained pipeline
pipeline = load_pipeline()

# make predictions
def predict(pipeline, X):

   

    # calculate the probability of the positive class
    y_proba = pipeline.predict_proba(X)[:, 1]  
    
    # apply threshold to get binary predictions
    y_pred = (y_proba >= config.THRESHOLD).astype(int)  
    
    return y_pred


# predict on new data
def predict_model(new_data):
    
    # validate input data
    if not isinstance(new_data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    
    # keep only the prediction columns
    new_data = new_data[config.RAW_INPUT_FEATURES]
    
    
    
    # make predictions
    predictions = predict(pipeline, new_data)
    return predictions
