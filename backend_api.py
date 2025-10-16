"""
================================================================================
BATTERY CAPACITY PREDICTION - BACKEND API
================================================================================

FastAPI backend for battery capacity and SOH prediction system.

ENDPOINTS:
1. POST /upload-training-data  - Upload multiple CSV files for training
2. POST /train                 - Train selected model on uploaded data
3. GET  /training-status       - Check training progress
4. POST /predict               - Predict capacity and SOH from NA file
5. GET  /models                - List all available trained models
6. GET  /model-info/{model}    - Get specific model information

FEATURES:
- 10 ML models to choose from
- Automatic label extraction from filenames
- Capacity + SOH% prediction
- Background training with status tracking
- Multiple model version support

USAGE:
1. Install: pip install -r requirements.txt
2. Run: uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
3. API Docs: http://localhost:8000/docs

================================================================================
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shutil
import time
from datetime import datetime
import json

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
TEMP_UPLOADS_DIR = "temp_uploads"
MODELS_DIR = "trained_models"
TRAINING_STATUS_FILE = "training_status.json"

# Create directories
Path(TEMP_UPLOADS_DIR).mkdir(exist_ok=True)
Path(MODELS_DIR).mkdir(exist_ok=True)

# Global training status
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "Ready",
    "model_name": None
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
MODEL_CONFIGS = {
    "Random Forest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 20, "random_state": 42, "n_jobs": -1}
    },
    "XGBoost": {
        "class": XGBClassifier if XGBOOST_AVAILABLE else None,
        "params": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1, "random_state": 42}
    },
    "Support Vector Machine": {
        "class": SVC,
        "params": {"kernel": "rbf", "C": 10, "gamma": "scale", "probability": True, "random_state": 42}
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 150, "max_depth": 10, "learning_rate": 0.1, "random_state": 42}
    },
    "K-Nearest Neighbors": {
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 5, "weights": "distance", "n_jobs": -1}
    },
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "params": {"max_depth": 20, "min_samples_split": 5, "random_state": 42}
    },
    "Neural Network (MLP)": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (100, 50), "max_iter": 500, "random_state": 42}
    },
    "LightGBM": {
        "class": LGBMClassifier if LIGHTGBM_AVAILABLE else None,
        "params": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1, "random_state": 42, "verbose": -1}
    },
    "CatBoost": {
        "class": CatBoostClassifier if CATBOOST_AVAILABLE else None,
        "params": {"iterations": 200, "depth": 10, "learning_rate": 0.1, "random_state": 42, "verbose": False}
    }
}

# Remove unavailable models
MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if v["class"] is not None}

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Battery Capacity Prediction API",
    description="Backend API for battery capacity and SOH prediction",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class TrainRequest(BaseModel):
    model_name: str

class TrainingResponse(BaseModel):
    status: str
    model_name: str
    train_accuracy: float
    test_accuracy: float
    training_time: str
    model_saved: str
    per_class_accuracy: dict

class PredictionResponse(BaseModel):
    capacity: str
    soh: float
    confidence: float

class ModelInfo(BaseModel):
    model_name: str
    trained_date: str
    train_accuracy: float
    test_accuracy: float
    file_path: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def extract_features(df):
    """Extract 19 statistical features from battery signature data."""
    features = {}
    
    # Frequency features (5)
    features['freq_mean'] = df['Frequency (Hz)'].mean()
    features['freq_std'] = df['Frequency (Hz)'].std()
    features['freq_min'] = df['Frequency (Hz)'].min()
    features['freq_max'] = df['Frequency (Hz)'].max()
    features['freq_median'] = df['Frequency (Hz)'].median()
    
    # Amplitude features (6)
    features['amp_mean'] = df['Amplitude S12 Sample 1'].mean()
    features['amp_std'] = df['Amplitude S12 Sample 1'].std()
    features['amp_min'] = df['Amplitude S12 Sample 1'].min()
    features['amp_max'] = df['Amplitude S12 Sample 1'].max()
    features['amp_median'] = df['Amplitude S12 Sample 1'].median()
    features['amp_range'] = features['amp_max'] - features['amp_min']
    
    # Phase features (6)
    features['phase_mean'] = df['Phase S12 Sample 1'].mean()
    features['phase_std'] = df['Phase S12 Sample 1'].std()
    features['phase_min'] = df['Phase S12 Sample 1'].min()
    features['phase_max'] = df['Phase S12 Sample 1'].max()
    features['phase_median'] = df['Phase S12 Sample 1'].median()
    features['phase_range'] = features['phase_max'] - features['phase_min']
    
    # Correlation features (3)
    features['freq_amp_corr'] = df['Frequency (Hz)'].corr(df['Amplitude S12 Sample 1'])
    features['freq_phase_corr'] = df['Frequency (Hz)'].corr(df['Phase S12 Sample 1'])
    features['amp_phase_corr'] = df['Amplitude S12 Sample 1'].corr(df['Phase S12 Sample 1'])
    
    return features

def extract_label_from_filename(filename):
    """
    Extract capacity label from filename.
    Examples:
    - "3500_mAh_Cell.csv" -> 3500
    - "4200_mAh_Cell_5.csv" -> 4200
    """
    try:
        # Find the capacity number in filename
        parts = filename.split('_')
        for part in parts:
            if part.isdigit() and 3500 <= int(part) <= 4900:
                return int(part)
    except:
        pass
    return None

def calculate_soh(predicted_capacity):
    """Calculate State of Health (SOH%) based on predicted capacity."""
    RATED_CAPACITY = 4900  # Maximum capacity
    soh = (predicted_capacity / RATED_CAPACITY) * 100
    return round(soh, 2)

def load_uploaded_data():
    """Load all CSV files from temp_uploads directory."""
    X = []
    y = []
    
    upload_path = Path(TEMP_UPLOADS_DIR)
    csv_files = list(upload_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError("No CSV files found in uploads directory")
    
    for csv_file in csv_files:
        try:
            # Extract label from filename
            label = extract_label_from_filename(csv_file.name)
            
            if label is None:
                print(f"Warning: Could not extract label from {csv_file.name}, skipping...")
                continue
            
            # Load and extract features
            df = pd.read_csv(csv_file)
            
            # Validate columns
            required_cols = ['Frequency (Hz)', 'Amplitude S12 Sample 1', 'Phase S12 Sample 1']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {csv_file.name} missing required columns, skipping...")
                continue
            
            features = extract_features(df)
            X.append(features)
            y.append(label)
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    if len(X) == 0:
        raise ValueError("No valid training data could be loaded")
    
    return pd.DataFrame(X), np.array(y)

def train_model_background(model_name: str):
    """Background task for training model."""
    global training_status
    
    try:
        start_time = time.time()
        
        # Update status
        training_status["is_training"] = True
        training_status["progress"] = 10
        training_status["message"] = "Loading training data..."
        training_status["model_name"] = model_name
        
        # Load data
        X, y = load_uploaded_data()
        training_status["progress"] = 30
        training_status["message"] = f"Loaded {len(X)} samples"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        training_status["progress"] = 40
        training_status["message"] = "Training model..."
        
        # Get model configuration
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not available")
        
        config = MODEL_CONFIGS[model_name]
        model_class = config["class"]
        params = config["params"]
        
        # Train model
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        training_status["progress"] = 70
        training_status["message"] = "Calculating accuracies..."
        
        # Calculate accuracies
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred) * 100
        test_acc = accuracy_score(y_test, test_pred) * 100
        
        # Per-class accuracy
        per_class_acc = {}
        for capacity in sorted(set(y_test)):
            mask = y_test == capacity
            if mask.sum() > 0:
                acc = accuracy_score(y_test[mask], test_pred[mask]) * 100
                per_class_acc[int(capacity)] = round(acc, 2)
        
        training_status["progress"] = 90
        training_status["message"] = "Saving model..."
        
        # Save model with metadata
        model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        model_path = Path(MODELS_DIR) / model_filename
        
        model_data = {
            "model": model,
            "model_name": model_name,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "per_class_accuracy": per_class_acc,
            "trained_date": datetime.now().isoformat(),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        joblib.dump(model_data, model_path)
        
        # Calculate training time
        training_time = time.time() - start_time
        training_time_str = f"{training_time / 60:.2f} minutes" if training_time > 60 else f"{training_time:.2f} seconds"
        
        # Update final status
        training_status["progress"] = 100
        training_status["message"] = "Training completed successfully!"
        training_status["result"] = {
            "status": "success",
            "model_name": model_name,
            "train_accuracy": round(train_acc, 2),
            "test_accuracy": round(test_acc, 2),
            "training_time": training_time_str,
            "model_saved": model_filename,
            "per_class_accuracy": per_class_acc
        }
        
        # Clean up uploaded files after successful training
        for file in Path(TEMP_UPLOADS_DIR).glob("*.csv"):
            file.unlink()
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["progress"] = 0
        training_status["message"] = f"Training failed: {str(e)}"
        training_status["result"] = {"status": "error", "message": str(e)}
        
    finally:
        training_status["is_training"] = False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "message": "Battery Capacity Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload-training-data",
            "train": "/train",
            "status": "/training-status",
            "predict": "/predict",
            "models": "/models"
        },
        "docs": "/docs"
    }

@app.post("/upload-training-data")
async def upload_training_data(files: List[UploadFile] = File(...)):
    """
    Upload multiple CSV files for training.
    Files should have naming convention: 3500_mAh_Cell.csv, 3600_mAh_Cell.csv, etc.
    """
    try:
        # Clear previous uploads
        for file in Path(TEMP_UPLOADS_DIR).glob("*.csv"):
            file.unlink()
        
        uploaded_files = []
        skipped_files = []
        
        for file in files:
            if not file.filename.endswith('.csv'):
                skipped_files.append(f"{file.filename} (not a CSV)")
                continue
            
            # Save file
            file_path = Path(TEMP_UPLOADS_DIR) / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(file.filename)
        
        return {
            "status": "success",
            "message": f"Uploaded {len(uploaded_files)} CSV files",
            "uploaded_files": uploaded_files,
            "skipped_files": skipped_files,
            "ready_for_training": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/train")
async def train_model_endpoint(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start training the selected model on uploaded data.
    Training runs in background and status can be checked via /training-status.
    """
    global training_status
    
    # Check if already training
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Check if model exists
    if request.model_name not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model_name}' not available. Available models: {available_models}"
        )
    
    # Check if files uploaded
    csv_files = list(Path(TEMP_UPLOADS_DIR).glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=400, detail="No training data uploaded. Please upload CSV files first.")
    
    # Reset training status
    training_status = {
        "is_training": True,
        "progress": 0,
        "message": "Starting training...",
        "model_name": request.model_name,
        "result": None
    }
    
    # Start training in background
    background_tasks.add_task(train_model_background, request.model_name)
    
    return {
        "status": "started",
        "message": f"Training {request.model_name} started in background",
        "model_name": request.model_name,
        "check_status_at": "/training-status"
    }

@app.get("/training-status")
async def get_training_status():
    """
    Get current training status and progress.
    Poll this endpoint to check if training is complete.
    """
    return training_status

@app.post("/predict")
async def predict_capacity(file: UploadFile = File(...)):
    """
    Predict battery capacity and SOH% from uploaded NA file.
    Returns predicted capacity (mAh) and State of Health (%).
    """
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Find the latest trained model (you can modify to select specific model)
        model_files = list(Path(MODELS_DIR).glob("*.pkl"))
        if not model_files:
            raise HTTPException(status_code=400, detail="No trained model found. Please train a model first.")
        
        # Load the most recent model
        latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
        model_data = joblib.load(latest_model_file)
        model = model_data["model"]
        
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Validate columns
        required_cols = ['Frequency (Hz)', 'Amplitude S12 Sample 1', 'Phase S12 Sample 1']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_cols}"
            )
        
        # Extract features
        features = extract_features(df)
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        probabilities = model.predict_proba(feature_df)[0]
        confidence = max(probabilities) * 100
        
        # Calculate SOH
        soh = calculate_soh(prediction)
        
        return {
            "capacity": f"{int(prediction)} mAh",
            "soh": soh,
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """
    List all available trained models with their information.
    """
    try:
        model_files = list(Path(MODELS_DIR).glob("*.pkl"))
        
        if not model_files:
            return {
                "status": "success",
                "models": [],
                "message": "No trained models found"
            }
        
        models_info = []
        for model_file in model_files:
            try:
                model_data = joblib.load(model_file)
                models_info.append({
                    "model_name": model_data.get("model_name", "Unknown"),
                    "trained_date": model_data.get("trained_date", "Unknown"),
                    "train_accuracy": model_data.get("train_accuracy", 0),
                    "test_accuracy": model_data.get("test_accuracy", 0),
                    "file_path": str(model_file)
                })
            except:
                continue
        
        return {
            "status": "success",
            "models": models_info,
            "count": len(models_info)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/model-info/{model_name}")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific trained model.
    """
    try:
        model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        model_path = Path(MODELS_DIR) / model_filename
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        model_data = joblib.load(model_path)
        
        return {
            "status": "success",
            "model_name": model_data.get("model_name", "Unknown"),
            "trained_date": model_data.get("trained_date", "Unknown"),
            "train_accuracy": model_data.get("train_accuracy", 0),
            "test_accuracy": model_data.get("test_accuracy", 0),
            "per_class_accuracy": model_data.get("per_class_accuracy", {}),
            "training_samples": model_data.get("training_samples", 0),
            "test_samples": model_data.get("test_samples", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/available-models")
async def get_available_models():
    """
    Get list of all ML models available for training.
    """
    return {
        "status": "success",
        "models": list(MODEL_CONFIGS.keys()),
        "count": len(MODEL_CONFIGS)
    }

@app.delete("/clear-uploads")
async def clear_uploads():
    """
    Clear all uploaded CSV files from temp_uploads directory.
    """
    try:
        count = 0
        for file in Path(TEMP_UPLOADS_DIR).glob("*.csv"):
            file.unlink()
            count += 1
        
        return {
            "status": "success",
            "message": f"Cleared {count} uploaded files"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear uploads: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Run on API startup."""
    print("="*70)
    print("Battery Capacity Prediction API Started")
    print("="*70)
    print(f"Temp uploads directory: {TEMP_UPLOADS_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Available models: {len(MODEL_CONFIGS)}")
    print("API Documentation: http://localhost:8000/docs")
    print("="*70)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)