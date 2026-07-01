# import necessary files and libraries
import os
import pathlib
import fraud_model
from fraud_model.data_manager import create_train_test_data, save_pipeline
from fraud_model.preprocessing import preprocess_dataset
from fraud_model.pipeline import create_pipeline
from fraud_model import config

# create training pipeline
def train_pipeline():
    
    print("Creating train/test datasets...")
    # Load the training data
    train_df, _ = create_train_test_data()
    
    print("Preprocessing training data...")
    # Preprocess the training data
    train_df = preprocess_dataset(train_df)
    
    # split the features and target variable
    X_train = train_df.drop(columns=[config.TARGET_FEATURE])
    y_train = train_df[config.TARGET_FEATURE]
    
    print("Creating pipeline...")

    # Create the pipeline
    pipeline = create_pipeline()
    
    print("Training model...")

    # train the model
    pipeline.fit(X_train, y_train)
    
    print("Saving trained pipeline...")
    # save pipeline to disk
    save_pipeline(pipeline)
    
    print("Training complete.")
    return pipeline


if __name__ == "__main__":
    train_pipeline()
