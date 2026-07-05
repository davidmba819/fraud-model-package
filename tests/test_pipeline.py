from sklearn.pipeline import Pipeline   
from fraud_model.pipeline import create_pipeline

# Test the create_pipeline function
def test_create_pipeline():
    pipeline = create_pipeline()
    
    # Check if the returned object is an instance of sklearn's Pipeline
    assert isinstance(pipeline, Pipeline), "The returned object is not a Pipeline instance."
    
    