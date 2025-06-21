import modal
from typing import Any, Dict, List
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import pickle
import json
import os
import clip

# Define the Modal app
app = modal.App("fashion-recognition-api")

# Create a Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torchvision", 
    "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1",
    "pillow",
    "numpy",
    "fastapi",
    "python-multipart",
    "uvicorn"
)

# Create a volume to store model files and embeddings
volume = modal.Volume.from_name("fashion-models", create_if_missing=True)

# Global variables for models and data (will be loaded once per container)
model = None
preprocess = None
product_index = None
designer_metadata = None

@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="T4",  # Use GPU for faster inference
    timeout=300,
    memory=8192  # 8GB RAM for model loading
)
def load_models():
    """Load CLIP model and reference data into memory."""
    global model, preprocess, product_index, designer_metadata
    
    print("--- Loading models and data ---")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model 'ViT-B/32'...")
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        print("✅ CLIP model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP model: {e}")

    # Load the indexed product embeddings
    index_file = "/models/reference_embeddings.pkl"
    print(f"Loading product index from '{index_file}'...")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"'{index_file}' not found. Please upload the embeddings file.")
    with open(index_file, "rb") as f:
        product_index = pickle.load(f)
    print(f"✅ Loaded {len(product_index)} product embeddings")
        
    # Load designer metadata
    metadata_file = "/models/designer-metadata.json"
    print(f"Loading designer metadata from '{metadata_file}'...")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"'{metadata_file}' not found.")
    with open(metadata_file, "r") as f:
        designer_metadata = json.load(f)
    print(f"✅ Loaded metadata for {len(designer_metadata)} products")
        
    print("--- All models and data loaded successfully ---")
    return True

@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="T4",
    timeout=60,
    memory=4096
)
def get_image_embedding(image_bytes: bytes) -> np.ndarray:
    """Generates a vector embedding for an image from bytes."""
    global model, preprocess
    
    # Load models if not already loaded
    if model is None:
        load_models.remote()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Process image
    processed_image = preprocess(image).unsqueeze(0).to(device)
    
    # Generate embedding
    with torch.no_grad():
        embedding = model.encode_image(processed_image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy()

@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="T4",
    timeout=60,
    memory=4096
)
def classify_outfit(image_bytes: bytes, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Classify an uploaded outfit image and return the top N most similar products
    with their metadata.
    """
    global product_index, designer_metadata
    
    # Load models and data if not already loaded
    if product_index is None or designer_metadata is None:
        load_models.remote()
    
    try:
        # Generate embedding for the uploaded image
        query_embedding = get_image_embedding.remote(image_bytes)
        
        # Prepare reference embeddings for search
        reference_embeddings = np.vstack([item['embedding'] for item in product_index])

        # Compute cosine similarity
        similarities = np.dot(reference_embeddings, query_embedding.T).flatten()

        # Find the top N best matches
        top_n_positions = np.argsort(similarities)[-top_n:][::-1]

        # Build the response
        results = []
        for position in top_n_positions:
            # Get the indexed item
            indexed_item = product_index[position]
            product_folder_path = indexed_item['product_folder']
            
            # Derive the string key for the metadata lookup
            product_folder_name = os.path.basename(product_folder_path)
            metadata_key = product_folder_name.replace('-', '_')
            
            # Get metadata
            metadata = designer_metadata.get(metadata_key, {})
            results.append({
                "confidence": float(similarities[position]),
                "outfit_url": metadata.get("checkout_link", "URL not found"),
                "product_name": metadata_key
            })
            
        return results

    except Exception as e:
        raise RuntimeError(f"Classification failed: {str(e)}")

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=30,
    memory=1024
)
def health_check() -> Dict[str, Any]:
    """Health check endpoint for deployment monitoring."""
    try:
        # Try to load models to check if everything is working
        load_models.remote()
        return {
            "status": "healthy",
            "models_loaded": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "product_count": len(product_index) if product_index else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "models_loaded": False
        }

@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("🧪 Testing Modal Fashion Recognition API...")
    
    # Test health check
    print("🏥 Testing health check...")
    health = health_check.remote()
    print(f"Health status: {health}")
    
    # Test with a sample image (you'll need to provide one)
    print("\n📸 To test classification, run:")
    print("python modal_fashion_api.py classify <path_to_image>")
    
    return health

@app.function()
def classify(image_path: str, top_n: int = 5):
    """Local entrypoint for testing classification."""
    print(f"🔍 Classifying image: {image_path}")
    
    # Read image file
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Perform classification
    results = classify_outfit.remote(image_bytes, top_n)
    
    print(f"\n🎯 Top {top_n} matches:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['product_name']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   URL: {result['outfit_url']}")
        print()
    
    return results

# For web deployment, you can create a FastAPI wrapper
@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="T4",
    timeout=300,
    memory=8192
)
@modal.fastapi_endpoint(method="POST")
async def classify_web(image: bytes, top_n: int = 5):
    """Web endpoint for image classification."""
    try:
        results = classify_outfit.remote(image, top_n)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=30,
    memory=1024
)
@modal.fastapi_endpoint(method="GET")
async def health_web():
    """Web endpoint for health check."""
    return health_check.remote() 