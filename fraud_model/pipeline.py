import pandas as pd
import numpy as np
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from fraud_model import config
from fraud_model.preprocessing import Fraudprocessor


# create preprocessing for transformation
def create_preprocessor():
    # Initialize the OneHotEncoder
    encoder = ohe(sparse_output=False, handle_unknown='ignore', drop='first')  # drop='first' to avoid multicollinearity

# Create a ColumnTransformer to apply the encoder to categorical features
    preprocessor = ColumnTransformer(transformers=[('cat', encoder, config.CAT_TO_ENCODE)],
                                     remainder='passthrough'  # Keep the rest of the columns unchanged
                                    )       
      
    return preprocessor

# create pipeline for model training
def create_pipeline():
    preprocessor = create_preprocessor()
   
    model = XGBClassifier(**config.XGBOOST_PARAMS)
    
    pipeline = Pipeline(steps=[
        ('fraud_preprocessor', Fraudprocessor()),
        ('column_transformer', preprocessor), 
        ('model', model)
        ]
                        )
    
    return pipeline

