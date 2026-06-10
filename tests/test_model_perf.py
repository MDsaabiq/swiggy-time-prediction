import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from mlflow import MlflowClient

dagshub.init(repo_owner='saabiqcs', repo_name='swiggy-time-prediction', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/saabiqcs/swiggy-time-prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# set model name
model_name = load_model_information("run_information.json")["model_name"]
stage = "Production"

# mlflow client
client = MlflowClient()

# --- ROBUST MODEL LOADING FALLBACK FOR TESTS ---
try:
    # 1. Fetch all available versions from the registry
    all_versions = client.get_latest_versions(name=model_name)
    
    if not all_versions:
        raise Exception(f"The model name '{model_name}' does not exist in your DagsHub registry at all!")

    # 2. Match target stage
    target_version = None
    for mv in all_versions:
        if mv.current_stage.strip().lower() == stage.strip().lower():
            target_version = mv.version
            break
            
    # 3. Fallback to latest version if stage is empty
    if not target_version:
        target_version = all_versions[0].version
        model_path = f"models:/{model_name}/{target_version}"
    else:
        model_path = f"models:/{model_name}/{target_version}"

    # 4. Load the latest model from model registry
    model = mlflow.sklearn.load_model(model_path)

except Exception as e:
    raise RuntimeError(f"CRITICAL TEST CONFIGURATION ERROR: Failed to load model from registry. Details: {e}")
# -----------------------------------------------

# set the root path
root_path = Path(__file__).parent.parent

# load the preprocessor
preprocessor_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)


# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ("regressor", model)
])

test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe, test_data_path, threshold_error",
                        argvalues=[(model_pipe, test_data_path, 5)])
def test_model_performance(model_pipe, test_data_path, threshold_error):
    # load test data
    df = pd.read_csv(test_data_path)
    
    # drop the missing values
    df.dropna(inplace=True)
    
    # make X and y
    X = df.drop(columns=["time_taken"])
    y = df['time_taken']
    
    # get the predictions
    y_pred = model_pipe.predict(X)
    
    # calculate the mean error
    mean_error = mean_absolute_error(y, y_pred)
    
    # check for performance
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes"
    print("The avg error is", mean_error)
    
    print(f"The {model_name} model passed the performance test")