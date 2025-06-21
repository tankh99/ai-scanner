from typing import Any, Dict, List
from io import BytesIO
from fastapi import FastAPI, HTTPException
from PIL import Image
import torch
import uvicorn
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import json
import base64
from ensemble_model import get_ensemble_model
from contextlib import asynccontextmanager

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Global ensemble model
ensemble_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ensemble_model
    print("--- Loading ensemble model ---")
    try:
        ensemble_model = get_ensemble_model()
        print("--- Ensemble model loaded successfully ---")
    except Exception as e:
        print(f"Warning: Failed to load ensemble model: {e}")
        print("Server will start with limited functionality")
    
    yield
    
    # Shutdown
    print("--- Shutting down ---")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify")
async def classify_outfit(image_data: Dict[str, str]):
    """
    Classify an outfit from base64 image string and return purchase link.
    
    Expected input:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns:
    {
        "purchase_link": "https://...",
        "confidence": 0.95,
        "model_used": "ResNet"
    }
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Server is not ready, models are loading.")

    try:
        # Extract base64 image string
        if "image" not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        base64_string = image_data["image"]
        
        # Remove data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        
        # Decode base64 to image
        try:
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Get ensemble prediction
        ensemble_result = ensemble_model.ensemble_predict(image)
        
        # Extract purchase link from primary prediction
        primary_prediction = ensemble_result["primary_prediction"]
        purchase_link = primary_prediction.get("outfit_url", "URL not found")
        
        # Build response
        response = {
            "purchase_link": purchase_link,
            "confidence": ensemble_result["ensemble_confidence"],
            "model_used": primary_prediction.get("model", "Unknown")
        }
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify model status."""
    return {
        "status": "healthy" if ensemble_model is not None else "unhealthy",
        "models_loaded": ensemble_model is not None
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
