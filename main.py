from typing import Any, Dict, List
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import uvicorn
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import json
from fastapi import HTTPException
import clip

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals for CLIP model and data ---
model = None
preprocess = None
product_index = None
designer_metadata = None

@app.on_event("startup")
def startup_event():
    """Load models and data into memory on application startup."""
    global model, preprocess, product_index, designer_metadata

    print("--- Loading models and data ---")
    
    # Load CLIP model
    print(f"Loading CLIP model 'ViT-B/32' on device '{DEVICE}'...")
    try:
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP model: {e}")

    # Load the indexed product embeddings
    index_file = "reference_embeddings.pkl"
    print(f"Loading product index from '{index_file}'...")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"'{index_file}' not found. Please run the indexing script first.")
    with open(index_file, "rb") as f:
        product_index = pickle.load(f)
        
    # Load designer metadata
    metadata_file = "designer-metadata.json"
    print(f"Loading designer metadata from '{metadata_file}'...")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"'{metadata_file}' not found.")
    with open(metadata_file, "r") as f:
        designer_metadata = json.load(f)
        
    print("--- Models and data loaded successfully ---")


def get_image_embedding(image: Image.Image) -> np.ndarray:
    """Generates a vector embedding for a single PIL image."""
    global model, preprocess, DEVICE
    
    processed_image = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(processed_image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

@app.post("/classify")
async def classify_outfit(file: UploadFile = File(...), top_n: int = 5):
    """
    Classify an uploaded outfit image and return the top N most similar products
    with their metadata.
    """
    if not all([model, preprocess, product_index, designer_metadata]):
        raise HTTPException(status_code=503, detail="Server is not ready, models are loading.")

    try:
        # Read uploaded image into a PIL Image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Generate embedding for the uploaded image
        query_embedding = get_image_embedding(image)
        
        # Prepare reference embeddings for search
        reference_embeddings = np.vstack([item['embedding'] for item in product_index])

        # Compute cosine similarity
        similarities = np.dot(reference_embeddings, query_embedding.T).flatten()

        # Find the top N best matches
        # argsort returns the *indices* (positions) of the elements in sorted order
        top_n_positions = np.argsort(similarities)[-top_n:][::-1]

        # Build the response
        results = []
        for position in top_n_positions:
            # 1. Use the numeric position to get the corresponding item from our index list
            indexed_item = product_index[position]
            product_folder_path = indexed_item['product_folder']
            
            # 2. Derive the string key for the metadata lookup from the folder path
            product_folder_name = os.path.basename(product_folder_path)
            metadata_key = product_folder_name.replace('-', '_')
            
            # 3. Use the correct string key to look up the metadata
            metadata = designer_metadata.get(metadata_key, {})
            results.append({
                "confidence": float(similarities[position]),
                "outfit_url": metadata.get("checkout_link", "URL not found")
            })
            
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 4000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
