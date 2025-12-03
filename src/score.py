import json
import joblib
import numpy as np
import os

def init():
    """
    This function is called when the container is initialized/started.
    It loads the model into memory so we don't have to reload it for every request.
    """
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    
    print(f"Scanning for model in: {model_dir}")
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith("rul_model.pkl"):
                model_path = os.path.join(root, file)
                break
    
    if model_path is None:
        raise FileNotFoundError(f"Could not find 'rul_model.pkl' in {model_dir}")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

def run(raw_data):
    """
    This function runs for every API call.
    It receives JSON data, parses it, predicts, and returns the result.
    """
    try:
        json_data = json.loads(raw_data)
        if "data" in json_data:
            input_data = np.array(json_data["data"])
        else:
            input_data = np.array(json_data)
        predictions = model.predict(input_data)
        return {"result": predictions.tolist()}
    
    except Exception as e:
        return {"error": str(e)}