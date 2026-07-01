# import libraries
import os
import pathlib
import joblib
import fraud_model
import pandas as pd
import numpy as np
from fraud_model import config
from sklearn.model_selection import train_test_split

# loading the dataset
def load_dataset():
    """Load the raw dataset with optimized data types."""
    
    df =pd.read_csv(os.path.join(config.RAW_DATA_PATH, config.RAW_DATA_FILE), dtype=config.MEMORY_DTYPES)
    return df


# sample dataset
def sample_dataset(df):
    """ Sample the dataset to reduce it to 1 million rows while
    preserving the class distribution."""
    
    sampled_df, _ = train_test_split(df, 
                                  train_size=1_000_000, 
                                  random_state=config.RANDOM_SEED, 
                                  stratify=df[config.TARGET_FEATURE])
    return sampled_df

# split dataset into train and test sets
def split_dataset(df):
    """Split the dataset into train and test sets."""
    
    train_df, test_df = train_test_split(df, 
                                         test_size=config.TEST_SIZE, 
                                         random_state=config.RANDOM_SEED, 
                                         stratify=df[config.TARGET_FEATURE])
    return train_df, test_df

# save train and test datasets to csv files
def save_train_test_data(train_df, test_df):
    """Save the train and test datasets to csv files."""
    
    train_df.to_csv(os.path.join(config.PROCESSED_DATA_PATH, config.TRAIN_FILE), index=False)
    test_df.to_csv(os.path.join(config.PROCESSED_DATA_PATH, config.TEST_FILE), index=False)

# load train dataset
def load_train_data():
    """Load the train dataset."""
    
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, config.TRAIN_FILE), dtype=config.MEMORY_DTYPES)
    return train_df

# load test dataset
def load_test_data():
    """Load the test dataset."""
    
    test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, config.TEST_FILE), dtype=config.MEMORY_DTYPES)
    
    return test_df

# save pipeline to disk
def save_pipeline(pipeline):
    """Save the trained machine learning pipeline."""
    
    joblib.dump(pipeline, os.path.join(config.MODEL_PATH, config.MODEL_NAME))

# load pipeline from disk
def load_pipeline():
    """Load the trained machine learning pipeline."""
    
    pipeline = joblib.load(os.path.join(config.MODEL_PATH, config.MODEL_NAME))
    return pipeline


# create train and test datasets from the raw dataset
def create_train_test_data():
    """Create train and test datasets from the raw dataset."""
    
    # load the raw dataset
    raw_df = load_dataset()
    
    # sample the dataset to reduce it to 1 million rows
    sampled_df = sample_dataset(raw_df)
    
    # split the sampled dataset into train and test sets
    train_df, test_df = split_dataset(sampled_df)
    
    # save the train and test datasets to csv files
    save_train_test_data(train_df, test_df)
    
    # return the train and test datasets
    return train_df, test_df