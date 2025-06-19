from typing import Any, Dict
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import uvicorn
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
from segment_anything import sam_model_registry, SamPredictor

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables for similarity search
reference_embeddings = None
reference_labels = None
feature_extractor = None
sam_predictor = None

def load_sam_model():
    """Load SAM model for query image segmentation"""
    global sam_predictor
    if sam_predictor is None:
        print("Loading SAM model for query segmentation...")
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        print("✅ SAM model loaded successfully!")

def segment_query_image(image_path):
    """Segment the main outfit in a query image using SAM"""
    load_sam_model()
    
    # Load and segment the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    
    # Use center point as prompt
    h, w, _ = image.shape
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    
    masks, scores, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    # Take the mask with highest score
    best_mask = masks[np.argmax(scores)]
    
    # Apply mask to get segmented image
    image_pil = Image.open(image_path).convert("RGB")
    np_img = np.array(image_pil)
    mask_3c = np.stack([best_mask]*3, axis=-1)
    segmented_img = np_img * mask_3c
    
    return Image.fromarray(segmented_img.astype(np.uint8))

def load_reference_embeddings():
    """Load pre-computed reference embeddings for similarity search"""
    global reference_embeddings, reference_labels, feature_extractor
    
    # Load pre-trained ResNet50 for feature extraction
    feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)
    
    # Load pre-computed embeddings if they exist
    embeddings_file = "reference_embeddings_segmented.pkl"
    labels_file = "reference_labels_segmented.json"
    
    if os.path.exists(embeddings_file) and os.path.exists(labels_file):
        print("Loading pre-computed segmented embeddings...")
        with open(embeddings_file, 'rb') as f:
            reference_embeddings = pickle.load(f)
        with open(labels_file, 'r') as f:
            reference_labels = json.load(f)
    else:
        print("Computing reference embeddings from segmented images...")
        compute_reference_embeddings()

def compute_reference_embeddings():
    """Compute embeddings for all segmented reference images"""
    global reference_embeddings, reference_labels, feature_extractor
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    reference_embeddings = []
    reference_labels = []
    
    base_dir = "./sam_segmented/sorted/"
    for designer in os.listdir(base_dir):
        designer_dir = os.path.join(base_dir, designer)
        if not os.path.isdir(designer_dir) or designer == "fraudfd":
            continue
        
        print(f"Processing {designer}...")
        for product in os.listdir(designer_dir):
            product_dir = os.path.join(designer_dir, product)
            if not os.path.isdir(product_dir):
                continue
            
            # Look for cropped (segmented) images
            for img in os.listdir(product_dir):
                if img.endswith('_crop.png'):
                    img_path = os.path.join(product_dir, img)
                    try:
                        # Load and preprocess segmented image
                        image = Image.open(img_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0).to(device)
                        
                        # Extract features
                        with torch.no_grad():
                            features = feature_extractor(image_tensor)
                            features = features.view(features.size(0), -1)
                            features = features.cpu().numpy()
                        
                        reference_embeddings.append(features[0])
                        reference_labels.append(f"{designer}/{product}")
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
    
    # Save embeddings for future use
    with open("reference_embeddings_segmented.pkl", 'wb') as f:
        pickle.dump(reference_embeddings, f)
    with open("reference_labels_segmented.json", 'w') as f:
        json.dump(reference_labels, f)
    
    print(f"Computed embeddings for {len(reference_embeddings)} segmented reference images")

def similarity_search(image_path):
    """Perform similarity search to find the best matching outfit"""
    global reference_embeddings, reference_labels, feature_extractor
    
    if reference_embeddings is None:
        load_reference_embeddings()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Segment the query image first
    segmented_image = segment_query_image(image_path)
    
    # Preprocess the segmented image
    image_tensor = transform(segmented_image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), -1)
        query_features = features.cpu().numpy()
    
    # Calculate cosine similarity with all reference embeddings
    similarities = cosine_similarity(query_features, reference_embeddings)[0]
    
    # Find the best match
    best_match_idx = np.argmax(similarities)
    best_match_label = reference_labels[best_match_idx]
    confidence = similarities[best_match_idx]
    
    return best_match_label, confidence

@app.post("/classify")
async def classify_outfit(file: UploadFile = File(...)):
    """
    Classify an uploaded outfit image and return the best match with purchase information.
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"uploaded_images/{file.filename}"
        os.makedirs("uploaded_images", exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load feature extractor
        feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
        feature_extractor.eval()
        feature_extractor.to(device)
        
        # Load reference embeddings
        with open("reference_embeddings_segmented.pkl", 'rb') as f:
            reference_embeddings = pickle.load(f)
        with open("reference_labels_segmented.json", 'r') as f:
            reference_labels = json.load(f)
        
        # Preprocess uploaded image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Segment the uploaded image
        segmented_image = segment_query_image(temp_path)
        image_tensor = transform(segmented_image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.view(features.size(0), -1)
            query_features = features.cpu().numpy()
        
        # Calculate similarities
        similarities = cosine_similarity(query_features, reference_embeddings)[0]
        
        # Find best match with lower threshold for better recall
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Lower threshold from 0.5 to 0.35 for better recall
        if best_similarity < 0.35:
            return {
                "error": "No confident match found",
                "best_match": reference_labels[best_idx],
                "confidence": float(best_similarity),
                "message": "The uploaded image doesn't match any known outfits with sufficient confidence."
            }
        
        best_match = reference_labels[best_idx]
        
        # Get outfit details
        outfit_details = get_outfit_details(best_match)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {
            "outfit_name": best_match,
            "confidence": float(best_similarity),
            "details": outfit_details
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

def get_outfit_details(outfit_name: str) -> Dict[str, Any]:
    """Get detailed information about an outfit"""
    # Parse designer and product from outfit_name (format: "designer/product")
    if "/" in outfit_name:
        designer, product = outfit_name.split("/", 1)
    else:
        designer = "Unknown"
        product = outfit_name
    
    # Define outfit details based on designer
    outfit_details = {
        "designer": designer,
        "product_name": product,
        "price_range": "Contact for pricing",
        "availability": "Available",
        "description": f"Exclusive {product} by {designer}",
        "contact_info": {
            "email": "info@trailblazerqueen.com",
            "phone": "+1 (555) 123-4567"
        }
    }
    
    # Add specific details for each designer
    if designer.lower() == "marinakarelyan":
        outfit_details.update({
            "website": "https://marinakarelian.com",
            "instagram": "@marinakarelian",
            "category": "Evening Wear & Gowns"
        })
    elif designer.lower() == "reaumonde":
        outfit_details.update({
            "website": "https://reaumonde.com",
            "instagram": "@reaumonde",
            "category": "Western Wear & Accessories"
        })
    elif designer.lower() == "gavelparis":
        outfit_details.update({
            "website": "https://gavelparis.com",
            "instagram": "@gavelparis",
            "category": "Runway & Couture"
        })
    
    return outfit_details

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>The Trailblazer Queen - AI Outfit Scanner</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            .upload-area { border: 2px dashed #fff; padding: 40px; margin: 20px 0; border-radius: 10px; }
            .result { margin: 20px 0; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👑 The Trailblazer Queen</h1>
            <h2>AI Outfit Scanner</h2>
            <p>Upload a photo of any outfit and discover where to buy it!</p>
            
            <div class="upload-area">
                <input type="file" id="imageInput" accept="image/*" style="display: none;">
                <button onclick="document.getElementById('imageInput').click()">Choose Image</button>
                <p>or drag and drop an image here</p>
            </div>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('imageInput').addEventListener('change', async function(e) {
                const file = e.target.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/classify', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    
                    if (result.error) {
                        resultDiv.innerHTML = `<h3>❌ ${result.error}</h3><p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>`;
                    } else {
                        resultDiv.innerHTML = `
                            <h3>✅ Match Found!</h3>
                            <p><strong>Outfit:</strong> ${result.outfit_name}</p>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Designer:</strong> ${result.details.designer}</p>
                            <p><strong>Category:</strong> ${result.details.category}</p>
                            <p><strong>Contact:</strong> ${result.details.contact_info.email}</p>
                        `;
                    }
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('result').innerHTML = '<h3>❌ Error processing image</h3>';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("🚀 Starting The Trailblazer Queen AI Scanner with SAM Segmentation")
    print("=" * 60)
    print("📁 Loading segmented reference embeddings...")
    load_reference_embeddings()
    print("✅ Ready to scan outfits!")
    print("🌐 Server starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 