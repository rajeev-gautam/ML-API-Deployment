from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Step 1: Create FastAPI app
app = FastAPI()

# Step 2: Load saved model
model = joblib.load("iris_model.pkl")

# Step 3: Define input format
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Step 4: Create prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    # Convert input into array
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    # Predict
    prediction = model.predict(input_data)

    # Convert prediction into readable form
    flower_names = ["Setosa", "Versicolor", "Virginica"]

    result = flower_names[prediction[0]]

    # Return JSON response
    return {
        "prediction": result
    }