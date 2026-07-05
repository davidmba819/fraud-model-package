import pandas as pd
import pytest
from fraud_model import config
import pytest
import os
from fraud_model.data_manager import ( load_dataset, sample_dataset, 
                                        split_dataset, save_train_test_data, 
                                        load_train_data, load_test_data)

# create a pytest fixture to load the dataset
@pytest.fixture(scope="module")
def dataset():
    df = load_dataset()
    return df

# test load dataset function
print("Running test_load_dataset()...")
def test_load_dataset(dataset):
    assert isinstance(dataset, pd.DataFrame)
    assert not dataset.empty

# test sample dataset function
print("Running test_sample_dataset()...")
def test_sample_dataset(dataset):
    sampled_df = sample_dataset(dataset)
    assert isinstance(sampled_df, pd.DataFrame)
    assert len(sampled_df) == 1_000_000
    original = dataset[config.TARGET_FEATURE].value_counts(normalize=True)
    sampled = sampled_df[config.TARGET_FEATURE].value_counts(normalize=True)

    assert abs(original[0] - sampled[0]) < 1e-6
    assert abs(original[1] - sampled[1]) < 1e-6

# test split dataset function
print("Running test_split_dataset()...")
def test_split_dataset(dataset):
    df = sample_dataset(dataset)
    train_df, test_df = split_dataset(df)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) + len(test_df) == len(df)
    assert len(train_df) > 0
    assert len(test_df) > 0
    
# tests for save_train_test_data function
def test_save_train_test_data(dataset):

    df = sample_dataset(dataset)
    train_df, test_df = split_dataset(df)

    save_train_test_data(train_df, test_df)

    train_path = os.path.join(config.PROCESSED_DATA_PATH, config.TRAIN_FILE)
    test_path = os.path.join(config.PROCESSED_DATA_PATH, config.TEST_FILE)

    # Check if the files exist
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    