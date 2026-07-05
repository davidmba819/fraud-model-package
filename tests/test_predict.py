import pytest
import numpy as np
from fraud_model.predict import predict

from fraud_model.predict import predict_model

# Test the predict_model function with invalid input
def test_predict_model_invalid_input():

    with pytest.raises(ValueError):
        predict_model([1, 2, 3])

# create a dummy pipeline class for testing
class DummyPipeline:

    def predict_proba(self, X):
        return np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4]
        ])

# Test the predict function with a dummy pipeline and input data
def test_predict():

    pipeline = DummyPipeline()

    X = np.zeros((3, 2))

    predictions = predict(pipeline, X)

    assert predictions.tolist() == [0, 1, 1]