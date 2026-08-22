import joblib 
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_price_model.joblib")
features = joblib.load("house_features.joblib")

class HouseFeatures(BaseModel):
  MedInc : float = Field(gt=0, description="Median Income of the colony")
  HouseAge : float = Field(gt = 0, description = "The age of the house")
  AveRooms : float = Field(gt=0, description="Average number of the Rooms")
  AveBedrms: float = Field(gt = 0, description="Average number of Bedrooms")
  Population: float = Field(gt=0, description="The population of the Area")
  AveOccup: float = Field(gt=0, description="The occupation of the people living there")
  Latitude: float = Field(ge=32, le=42, description="Latitude of the particular house")
  Longitude : float = Field(ge=-125, le=-114, description="The longitude of the particular house")

#home page

@app.get("/")
def home():
  return {
    "message":"This the home page",
    "status":"model is running successfully",
    "endpoint":"send POST request to /predict"
  }

## health directory

@app.get("/health")
def health():
  return{
    "status":"running",
    "health":"$39,000",
    "model":"RandomForestClassifier",
    "features":features
  }

##predicted directory

@app.post("/predict")
def predict(house:HouseFeatures):
  try:
    input_data=pd.DataFrame(
      [
        {
          "MedInc":house.MedInc,
          "HouseAge":house.HouseAge,
          "AveRooms":house.AveRooms,
          "AveBedrms":house.AveBedrms,
          "Population":house.Population,
          "AveOccup":house.AveOccup,
          "Latitude":house.Latitude,
          "Longitude":house.Longitude
        }
      ]
    )

    predicted = model.predict(input)[0]
    price_usd=predicted*100000

    return{
      "predicted_price":f"${price_usd:,.0f}",
      "predicted_error":f"${predicted:,.2f} hundred thousand dollars",
      "fidented price":f"${price_usd-39000:,.0f} to ${price_usd+39000:,.0f}"
    }
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"The prediction has failed: {str(e)}"
    )  

  