from fastapi import FastAPI # type: ignore
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn # type: ignore
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub
import dagshub
import mlflow.client

dagshub.init(repo_owner='saabiqcs', repo_name='swiggy-time-prediction', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/saabiqcs/swiggy-time-prediction.mlflow")

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# columns to preprocess in data
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]

nominal_cat_cols = ['weather', 'type_of_order', 'type_of_vehicle', "festival", "city_type", "is_weekend", "order_time_of_day"]

ordinal_cat_cols = ["traffic", "distance_type"]

# mlflow client
client = MlflowClient()

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

# stage of the model
stage = "Production"

# --- NEW ROBUST MODEL LOADING BLOCK ---
try:
    print(f"Querying DagsHub registry for model name: '{model_name}'...")
    
    # 1. Fetch ALL latest versions across ALL stages
    all_versions = client.get_latest_versions(name=model_name)
    
    if not all_versions:
        raise Exception(f"The model name '{model_name}' does not exist in your DagsHub registry at all!")
    
    print("Found the following versions in your registry:")
    for mv in all_versions:
        print(f"  -> Version: {mv.version} | Current Stage: '{mv.current_stage}'")
    
    # 2. Look for the version that matches your target stage
    target_version = None
    for mv in all_versions:
        if mv.current_stage.strip().lower() == stage.strip().lower():
            target_version = mv.version
            break
            
    # 3. Smart Fallback: If stage is empty, grab the absolute latest version available
    if not target_version:
        target_version = all_versions[0].version
        print(f"Warning: No model found in stage '{stage}'. Falling back to latest available version: {target_version}")
    else:
        print(f"Success: Found version {target_version} matching stage '{stage}'")

    model_path = f"models:/{model_name}/{target_version}"

    # 4. Load the model from the determined path
    print(f"Loading model from path: {model_path} ...")
    model = mlflow.sklearn.load_model(model_path)
    print("Model loaded into memory successfully!")

except Exception as e:
    import sys
    print(f"CRITICAL REGISTRY ERROR: {e}")
    sys.exit(1) # Stop the container safely to log the error cleanly
# ---------------------------------------

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ("regressor", model)
])

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        }, index=[0]
    )
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return predictions

if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)