from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from enum import Enum
from pydantic import BaseModel, ValidationError
import xgboost as xgb
import pandas as pd
import json
import os
import logging
from typing import Dict

# Configure logging to output to console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Pydantic Data Contract with Enums matching training dataset values
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe" 
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Global variables for model and metadata
model = None
label_encoder_classes = None
feature_names = None

def load_model_and_metadata():
    """Load the trained XGBoost model and metadata"""
    global model, label_encoder_classes, feature_names
    
    logger.info("Starting model loading process...")
    
    # Load XGBoost model
    model_path = "app/data/model.json"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info(f"XGBoost model successfully loaded from {model_path}")
    
    # Load metadata for consistent preprocessing
    metadata_path = "app/data/model_metadata.json"
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found at {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    label_encoder_classes = metadata['label_encoder_classes']
    feature_names = metadata['feature_names']
    
    logger.info(f"Model metadata loaded successfully")
    logger.info(f"Target classes: {label_encoder_classes}")
    logger.info(f"Feature names: {feature_names}")

def preprocess_features(penguin_data: PenguinFeatures) -> pd.DataFrame:
    """
    Apply one-hot encoding consistent with training process
    """
    logger.debug(f"Preprocessing input features: {penguin_data.dict()}")
    
    # Convert to DataFrame
    data_dict = penguin_data.dict()
    df = pd.DataFrame([data_dict])
    
    # Apply one-hot encoding to categorical features (sex, island)
    # Using drop_first=True to match training preprocessing
    df_encoded = pd.get_dummies(df, columns=['sex', 'island'], drop_first=True)
    
    # Ensure all expected features are present in correct order
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    logger.debug(f"Features after preprocessing: {df_encoded.columns.tolist()}")
    
    # Return features in same order as training
    return df_encoded[feature_names]

# Initialize FastAPI app
app = FastAPI(title="Penguin Species Prediction API")

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors and return HTTP 400 with clear error messages
    """
    error_details = []
    
    for error in exc.errors():
        field = error.get('loc', ['unknown'])[-1]  # Get the field name
        error_type = error.get('type', 'validation_error')
        error_msg = error.get('msg', 'Invalid value')
        
        if 'enum' in error_type:
            # Handle enum validation errors for sex and island
            if field == 'sex':
                valid_values = [sex.value for sex in Sex]
                logger.debug(f"Invalid sex value provided. Valid values: {valid_values}")
                error_details.append({
                    "field": field,
                    "error": f"Invalid sex value. Must be one of: {valid_values}"
                })
            elif field == 'island':
                valid_values = [island.value for island in Island]
                logger.debug(f"Invalid island value provided. Valid values: {valid_values}")
                error_details.append({
                    "field": field,
                    "error": f"Invalid island value. Must be one of: {valid_values}"
                })
            else:
                error_details.append({
                    "field": field,
                    "error": f"Invalid value for {field}"
                })
        else:
            # Handle other validation errors
            logger.debug(f"Validation error for field '{field}': {error_msg}")
            error_details.append({
                "field": field,
                "error": error_msg
            })
    
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Validation error",
            "errors": error_details
        }
    )

# Load model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup initiated")
    try:
        load_model_and_metadata()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        raise e

@app.post("/predict")
async def predict_penguin_species(penguin_data: PenguinFeatures) -> Dict:
    """
    Predict penguin species from input features
    
    Validates sex and island against training dataset values,
    applies consistent one-hot encoding, and returns prediction
    """
    logger.info("Prediction request received")
    
    if model is None:
        logger.error("Model not loaded - cannot make prediction")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Log the validated input
        logger.info(f"Making prediction for: species={penguin_data.sex.value}, island={penguin_data.island.value}")
        
        # Preprocess input with consistent one-hot encoding
        processed_features = preprocess_features(penguin_data)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        prediction_proba = model.predict_proba(processed_features)[0]
        
        # Convert prediction index to species name
        predicted_species = label_encoder_classes[prediction]
        confidence = float(max(prediction_proba))
        
        logger.info(f"Prediction successful: {predicted_species} (confidence: {confidence:.4f})")
        
        return {
            "predicted_species": predicted_species,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)