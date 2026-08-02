# import libraries
import os
import pathlib
import fraud_model
import pandas as pd
import numpy as np
from fraud_model import config
from fraud_model import data_manager
from sklearn.base import BaseEstimator, TransformerMixin

# validate dataset columns
def validate_dataset_columns(df, expected_columns):
    """Validate that all required columns exist."""

    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    return df

# check for missing values in the dataset
def check_missing_values(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Missing values found in the dataset:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found in the dataset.")
    
    return df

# check for duplicates in the dataset
def check_duplicates(df):
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print(f"Found {duplicate_rows} duplicate rows in the dataset.")
    else:
        print("No duplicate rows found in the dataset.")
    
    return df

# extract destination account type from the destination account name
def extract_destination_account_type(df):
    """Extract the destination account type from the destination account name."""
    
    df[config.ACCOUNT_TYPE_FEATURE] = df[config.ACCOUNT_TYPE_SOURCE].str[0]
    return df

# create new features for balance changes
def create_balance_change_features(df):
    
    """Create new features for balance changes."""
    
    df[config.SENDER_BALANCE_CHANGE] = df[config.OLD_ORIGIN_BALANCE] - df[config.NEW_ORIGIN_BALANCE]
    df[config.DESTINATION_BALANCE_CHANGE] = df[config.NEW_DESTINATION_BALANCE] - df[config.OLD_DESTINATION_BALANCE]
    
    return df

# drop unnecessary features from the dataset
def drop_unnecessary_features(df):
    """Drop unnecessary features from the dataset."""
    
    df = df.drop(columns=config.DROP_FEATURES, errors="ignore")
    return df

# validate dataframe
def validate_dataframe(df):
    """Validate the dataframe by checking for missing values, duplicates, and expected columns."""
    
    df = validate_dataset_columns(df, config.RAW_INPUT_FEATURES)
    df = check_missing_values(df)
    df = check_duplicates(df)
    
    return df

# preprocess the dataset
def preprocess_dataset(df):
    """Preprocess the dataset by validating the dataframe, extracting destination account type,
    creating balance change features, and dropping unnecessary features."""
    
    df = validate_dataframe(df)
    df = extract_destination_account_type(df)
    df = create_balance_change_features(df)
    df = drop_unnecessary_features(df)
    
    return df


# create an orchestration class for preprocessing
class Fraudprocessor(BaseEstimator, TransformerMixin):
    """A class for preprocessing the fraud dataset."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform the input dataframe by preprocessing it."""
        X = X.copy()
        X = preprocess_dataset(X)
        return X

