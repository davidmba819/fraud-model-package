from fastapi import FastAPI
from pydantic import BaseModel
from fraud_model.predict import predict_model
import pandas as pd
from fastapi import HTTPException
app = FastAPI()

class FraudRequest(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

@app.get("/")
def home():
    return {"message": "Welcome to the Fraud Detection API!"}

@app.post("/predict")
def predict_fraud(request: FraudRequest):
    
    try:
        # convert the request data to a dictionary
     request_dict = request.model_dump() 
    
        # convert the request data to a DataFrame
     input_df = pd.DataFrame([request_dict])
    
        # make predictions
     predictions = predict_model(input_df)
    
     prediction_class = int(predictions[0])  
    
        # Convert to int for better readability
     if prediction_class == 1:
        label= "Fraud"
     else:
        label = "Not Fraud"
        
        # return the predictions as a list
     return {"predictions": label,
            'class': prediction_class,
            'status': 'success'}
    
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

