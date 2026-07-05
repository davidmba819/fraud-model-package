import pandas as pd
import pytest
from fraud_model import config
from fraud_model.preprocessing import (validate_dataset_columns, 
                                         check_missing_values, 
                                         check_duplicates, 
                                         extract_destination_account_type, 
                                         create_balance_change_features, 
                                         drop_unnecessary_features)

# Test cases for the validate_dataset_columns function
def test_validate_dataset_columns_valid():
    df = pd.DataFrame({
        "step": [1],
        "type": ["TRANSFER"],
        "amount": [1000]
    })

    expected = ["step", "type", "amount"]

    result = validate_dataset_columns(df, expected)

    assert result.equals(df)

# test case for missing columns in the dataset
def test_validate_dataset_columns_missing():
    df = pd.DataFrame({
        "step": [1],
        "amount": [1000]
    })

    expected = ["step", "type", "amount"]

    with pytest.raises(ValueError):
        validate_dataset_columns(df, expected)
        
# Test check_missing_values()
def test_check_missing_values_no_missing(capsys):

    df = pd.DataFrame({
        "step": [1],
        "type": ["TRANSFER"],
        "amount": [1000]
    })

    result = check_missing_values(df)

    captured = capsys.readouterr()

    assert "No missing values found in the dataset." in captured.out
    assert result.equals(df)


def test_check_missing_values_with_missing(capsys):

    df = pd.DataFrame({
        "step": [1],
        "type": [None],
        "amount": [1000]
    })

    result = check_missing_values(df)

    captured = capsys.readouterr()

    assert "Missing values found in the dataset:" in captured.out
    assert result.equals(df)
    
 # Test check_duplicates()
def test_check_duplicates_no_duplicates(capsys):

    df = pd.DataFrame({
        "step": [1, 2],
        "type": ["TRANSFER", "CASH_OUT"],
        "amount": [1000, 2000]
    })

    result = check_duplicates(df)

    captured = capsys.readouterr()

    assert "No duplicate rows found in the dataset." in captured.out
    assert result.equals(df)


def test_check_duplicates_with_duplicates(capsys):

    df = pd.DataFrame({
        "step": [1, 1],
        "type": ["TRANSFER", "TRANSFER"],
        "amount": [1000, 1000]
    })

    result = check_duplicates(df)

    captured = capsys.readouterr()

    assert "Found 1 duplicate rows in the dataset." in captured.out
    assert result.equals(df)
    
# Test extract_destination_account_type()
def test_extract_destination_account_type():

    df = pd.DataFrame({
        config.ACCOUNT_TYPE_SOURCE: [
            "C123456",
            "M987654",
            "C111111"
        ]
    })

    result = extract_destination_account_type(df)

    assert config.ACCOUNT_TYPE_FEATURE in result.columns

    assert result[config.ACCOUNT_TYPE_FEATURE].tolist() == [
        "C",
        "M",
        "C"
    ]
    
# test balance change features
def test_create_balance_change_features():

    df = pd.DataFrame({
        config.OLD_ORIGIN_BALANCE: [1000, 5000],
        config.NEW_ORIGIN_BALANCE: [700, 4500],
        config.OLD_DESTINATION_BALANCE: [200, 100],
        config.NEW_DESTINATION_BALANCE: [500, 600]
    })

    result = create_balance_change_features(df)

    # Check that the new columns exist
    assert config.SENDER_BALANCE_CHANGE in result.columns
    assert config.DESTINATION_BALANCE_CHANGE in result.columns

    # Check the calculated values
    assert result[config.SENDER_BALANCE_CHANGE].tolist() == [300, 500]
    assert result[config.DESTINATION_BALANCE_CHANGE].tolist() == [300, 500]
    
# test drop_unnecessary_features
def test_drop_unnecessary_features():

    df = pd.DataFrame({
        "step": [1],
        "amount": [1000],
        "type": ["TRANSFER"],
        config.DROP_FEATURES[0]: ["dummy"],
        config.DROP_FEATURES[1]: ["dummy"],
        config.DROP_FEATURES[2]: [0],
    })

    result = drop_unnecessary_features(df)

    # Check dropped columns
    for column in config.DROP_FEATURES:
        assert column not in result.columns

    # Check remaining columns
    assert "step" in result.columns
    assert "amount" in result.columns
    assert "type" in result.columns