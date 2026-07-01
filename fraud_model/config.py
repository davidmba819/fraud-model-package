# Configuration file for the fraud detection project
# Import necessary libraries

import os 
import pathlib
import fraud_model

TARGET_FEATURE = 'isFraud'

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

PACKAGE_ROOT = pathlib.Path(fraud_model.__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

RAW_DATA_FILE = "paysim.csv"

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model")



# feature engineering parameters
DATASET_COLUMNS = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest',
                   'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']

RAW_INPUT_FEATURES  = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest',
            'oldbalanceDest', 'newbalanceDest']

TRAINING_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]

PREDICTION_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg","newbalanceOrig",
                      "nameDest", "oldbalanceDest", "newbalanceDest"]

RAW_CAT_FEATURES = ['type', 'nameOrig', 'nameDest']

CAT_TO_ENCODE = ['type', "destination_type"]

RAW_NUM_FEATURES = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                      'newbalanceDest']

ACCOUNT_TYPE_FEATURES = ["nameOrig", "nameDest"]

DROP_FEATURES= ["nameOrig", "nameDest","isFlaggedFraud"]

LEAKED_FEATURES = ["isFlaggedFraud"]

OLD_ORIGIN_BALANCE = "oldbalanceOrg"
NEW_ORIGIN_BALANCE = "newbalanceOrig"

OLD_DESTINATION_BALANCE = "oldbalanceDest"
NEW_DESTINATION_BALANCE = "newbalanceDest"

SENDER_BALANCE_CHANGE = "sender_balance_change"
DESTINATION_BALANCE_CHANGE = "destination_balance_change"

ACCOUNT_TYPE_SOURCE = "nameDest"

ACCOUNT_TYPE_FEATURE = "destination_type"

SELECTED_MODEL = "xgboost"

MEMORY_DTYPES = {
    "step": "int32",
    "type": "category",
    "nameOrig": "category",
    "nameDest": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


RANDOM_SEED = 42

TEST_SIZE = 0.20

STRATIFY = TARGET_FEATURE

MODEL_NAME = "xgboost_fraud_detector.pkl"

PREPROCESSOR_NAME = "preprocessor.pkl"

XGBOOST_PARAMS = {
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "subsample": 0.8,
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
}

THRESHOLD = 0.31