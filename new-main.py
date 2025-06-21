from io import BytesIO
from fastapi import File, UploadFile
import modal
import numpy as np
from PIL import Image
import torch


app = modal.App("example-get-started")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torchvision", 
    "ftfy",  # Required by CLIP
    "regex",  # Required by CLIP
    "clip",  # Use PyPI version instead of GitHub
    "pillow",
    "numpy",
    "fastapi",
    "python-multipart",
    "uvicorn"   
)

volume = modal.Volume.from_name("metadata", create_if_missing=True)


@app.function(image=image)
def get_image_embedding(image: Image.Image) -> np.ndarray:
    """Generates a vector embedding for a single PIL image."""
    global model, preprocess, DEVICE
    
    processed_image = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(processed_image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()


@app.function(image=image, volumes={"metadata": volume})
async def classify(file: UploadFile = File(...), top_n: int = 5):
    """
    Classify an uploaded outfit image and return the top N most similar products
    with their metadata.
    """
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
