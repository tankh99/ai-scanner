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
    embeddings_file = "reference_embeddings.pkl"
    labels_file = "reference_labels.json"
    
    if os.path.exists(embeddings_file) and os.path.exists(labels_file):
        print("Loading pre-computed embeddings...")
        with open(embeddings_file, 'rb') as f:
            reference_embeddings = pickle.load(f)
        with open(labels_file, 'r') as f:
            reference_labels = json.load(f)
    else:
        print("Computing reference embeddings...")
        compute_reference_embeddings()

def compute_reference_embeddings():
    """Compute embeddings for all reference images"""
    global reference_embeddings, reference_labels, feature_extractor
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    reference_embeddings = []
    reference_labels = []
    
    base_dir = "./enhanced_downloaded_images/"
    for outfit in os.listdir(base_dir):
        outfit_dir = os.path.join(base_dir, outfit)
        if not os.path.isdir(outfit_dir):
            continue
        
        for img in os.listdir(outfit_dir):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(outfit_dir, img)
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # Extract features
                    with torch.no_grad():
                        features = feature_extractor(image_tensor)
                        # Flatten the features
                        features = features.view(features.size(0), -1)
                        features = features.cpu().numpy()
                    
                    reference_embeddings.append(features[0])  # Remove batch dimension
                    reference_labels.append(outfit)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
    
    # Save embeddings for future use
    with open("reference_embeddings.pkl", 'wb') as f:
        pickle.dump(reference_embeddings, f)
    with open("reference_labels.json", 'w') as f:
        json.dump(reference_labels, f)
    
    print(f"Computed embeddings for {len(reference_embeddings)} reference images")

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
    
    # Load and preprocess the query image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
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
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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
        
        image = Image.open(temp_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
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
                "suggestion": "Try uploading a clearer image or different angle"
            }
        
        best_match = reference_labels[best_idx]
        
        print(f"Best Match {best_match}")
        # Get outfit details from database
        outfit_details = get_outfit_details(best_match)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {
            "outfit_name": outfit_details["name"],
            "confidence": float(best_similarity),
            "designer": outfit_details["designer"],
            "price": outfit_details["price"],
            "image_url": outfit_details["image_url"],
            "outfit_url": outfit_details["outfit_url"]
    }
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
def get_outfit_details(outfit_name: str) -> Dict[str, Any]:
    """
    Get additional details about an outfit.
    You'll need to implement this based on your data structure.
    This could be from a database, JSON file, or other source.
    """
    # Example implementation - replace with your actual data source
    outfit_data = {
        # Marina Karelyan
        "black-cut-out-velvet-gown-with-red-crystal-embellishments": {
            "designer": "Marina Karelian",
            "price": 2300,
            "name": "Black Cut-Out Velvet Gown with Red Crystal Embellishments",
            "outfit_url": "https://marinakarelian.com/products/black-cut-out-velvet-gown-with-red-crystal-embellishments",
            "image_Url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750163985/2cdacb22467e44f093a3b10b679ff0e6.thumbnail.0000000000_2048x_cciwpa.jpg"
        },
        "elegant-black-velvet-midi-dress-with-lace-detail": {
            "designer": "Marina Karelian",
            "price": 600,
            "name": "Elegant Black Velvet Midi Dress with Lace Detail",
            "outfit_url": "https://marinakarelian.com/products/elegant-black-velvet-midi-dress-with-lace-detail",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750163997/IMG_50812_nxszvc.jpg"
        },
        "elegant-silver-satin-cut-out-evening-gown-with-embellishments": {
            "designer": "Marina Karelian",
            "price": 1200,
            "name": "Elegant Silver Satin Cut-Out Evening Gown with Embellishments",
            "outfit_url": "https://marinakarelian.com/products/elegant-silver-satin-cut-out-evening-gown-with-embellishments",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164060/0A6FDF67-3AB1-4D65-8DB5-5BFAAD799F3B_wsgch5.jpg"
        },
        "emerald-satin-gown-with-beaded-neckline-and-waist": {
            "designer": "Marina Karelian",
            "price": 1400,
            "name": "Emerald Satin Gown with Beaded Neckline and Waist",
            "outfit_url": "https://marinakarelian.com/products/emerald-satin-gown-with-beaded-neckline-and-waist",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164081/DSC_3307_ssaffb.jpg"
        },
        "ice-blue-velvet-gown-with-crystal-embellishments": {
            "designer": "Marina Karelian",
            "price": 1500,
            "name": "Ice Blue Velvet Gown With Crystal Embellishments",
            "outfit_url": "https://marinakarelian.com/products/ice-blue-velvet-gown-with-crystal-embellishments",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164087/IMG_5087_xndtou.jpg"
        },
        "plum-velvet-cut-out-gown-with-embellished-sleeves": {
            "designer": "Marina Karelian",
            "price": 2800,
            "name": "Plum Velvet Cut-Out Gown with Embellished Sleeves",
            "outfit_url": "https://marinakarelian.com/products/plum-velvet-cut-out-gown-with-embellished-sleeves",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164109/IMG_5080_cpqqpu.jpg"
        },
        "sophisticated-black-velvet-bodycon-dress-with-red-embellishments": {
            "designer": "Marina Karelian",
            "price": 490,
            "name": "Sophisticated Black Velvet Bodycon Dress with Red Embellishments",
            "outfit_url": "https://marinakarelian.com/products/sophisticated-black-velvet-bodycon-dress-with-red-embellishments",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164128/IMG_7326_b6dc6257-2987-4f0a-9172-6fcfe5045496_xlpauz.jpg"
        },
        "silver-shimmer-midi-dress-with-bow-detail": {
            "designer": "Marina Karelian",
            "price": 290,
            "name": "Silver Shimmer Midi Dress with Bow Detail",
            "outfit_url": "https://marinakarelian.com/products/silver-shimmer-midi-dress-with-bow-detail",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164120/687c84eb402748f9ab5c162ab3862954.thumbnail.0000000000_2048x_uujsrk.jpg"
        },
        # Reaumonde
        "untitled-may13_10-18": {
            "designer": "Reau Monde",
            "price": 110,
            "name": "Fan Gear (Astros)",
            "outfit_url": "https://reaumonde.com/products/untitled-may13_10-18",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750165031/0AC8554A-D69B-4EC2-B7D6-D486F034DF86_bg6sul.png"
        },
        "denim-wrangler-cowboy-hat": {
            "designer": "Reau Monde",
            "price": 250,
            "name": "Monde Wrangler Cowboy Hat",
            "outfit_url": "https://reaumonde.com/products/denim-wrangler-cowboy-hat",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750165041/DSC03948_tyxnhh.png"
        },
        "monde-wrangler-cowboy-hat": {
            "designer": "Reau Monde",
            "price": 250,
            "name": "Monde Wrangler Cowboy Hat - Custom Green",
            "outfit_url": "https://reaumonde.com/products/monde-wrangler-cowboy-hat",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750165046/50D088C8-92E5-4876-BBD0-0F9358A0240A_oclvuq.png"
        },
        "rm-livestock-rodeo-tee": {
            "designer": "Reau Monde",
            "price": 65,
            "name": "RM 'Livestock' Rodeo Tee",
            "outfit_url": "https://reaumonde.com/products/rm-livestock-rodeo-tee",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750165010/DSC03757_q8xfx1.jpg"
        },
        "the-honey-comb-fedora": {
            "designer": "Reau Monde",
            "price": 275,
            "name": "The Honey Comb - Fedora",
            "outfit_url": "https://reaumonde.com/products/the-honey-comb-fedora",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750165060/C68AFA30-067A-4A5C-A3C7-7F2141917DF4_awalim.png"
        },
        "the-jones-fedora": {
            "designer": "Reau Monde",
            "price": 275,
            "name": "The Jones - Fedora",
            "outfit_url": "https://reaumonde.com/products/the-jones-fedora",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750165025/20A08394-09AD-4D8A-958F-DBA4433C549B_nakgvx.jpg"
        },
        # Fraud FD
        "fraud-c": {
            "designer": "Fraud FD",
            "price": 150,
            "name": "Fraud C Collection",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/fraud-c",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164000/fraud-c.jpg"
        },
        "fraud-c2": {
            "designer": "Fraud FD",
            "price": 180,
            "name": "Fraud C2 Collection",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/fraud-c2",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164005/fraud-c2.jpg"
        },
        "red-crystal-embellishment-zip-up-sweatshirt-with-pants": {
            "designer": "Fraud FD",
            "price": 120,
            "name": "Red Crystal Embellishment Zip-Up Sweatshirt with Pants",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/red-crystal-embellishment-zip-up-sweatshirt-with-pants",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164010/red-crystal-sweatshirt.jpg"
        },
        "white-t-shirt-with-printing-on-collar": {
            "designer": "Fraud FD",
            "price": 45,
            "name": "White T-Shirt with Printing on Collar",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/white-t-shirt-with-printing-on-collar",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164015/white-tshirt-collar.jpg"
        },
        "long-sleeve-stone-embellished-all-over-dress": {
            "designer": "Fraud FD",
            "price": 95,
            "name": "Long Sleeve Stone Embellished All Over Dress",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/long-sleeve-stone-embellished-all-over-dress",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164020/long-sleeve-dress.jpg"
        },
        "white-t-shirt": {
            "designer": "Fraud FD",
            "price": 35,
            "name": "White T-Shirt",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/white-t-shirt",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164025/white-tshirt.jpg"
        },
        "open-jogger-zip-track-jacket-set-rhinestone-embellishment": {
            "designer": "Fraud FD",
            "price": 85,
            "name": "Open Jogger Zip Track Jacket Set with Rhinestone Embellishment",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/open-jogger-zip-track-jacket-set-rhinestone-embellishment",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164030/track-jacket.jpg"
        },
        "crystal-all-over-dress-green": {
            "designer": "Fraud FD",
            "price": 110,
            "name": "Crystal All Over Dress - Green",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/crystal-all-over-dress-green",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164035/crystal-dress-green.jpg"
        },
        "orchid-patterned-bodycon-full-dress-white": {
            "designer": "Fraud FD",
            "price": 75,
            "name": "Orchid Patterned Bodycon Full Dress - White",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/orchid-patterned-bodycon-full-dress-white",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164040/orchid-dress-white.jpg"
        },
        "black-t-shirt-embroidary": {
            "designer": "Fraud FD",
            "price": 55,
            "name": "Black T-Shirt with Embroidery",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/black-t-shirt-embroidary",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164045/black-tshirt-embroidery.jpg"
        },
        "black-crystal-dresses": {
            "designer": "Fraud FD",
            "price": 130,
            "name": "Black Crystal Dresses",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/black-crystal-dresses",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164050/black-crystal-dresses.jpg"
        },
        "crew-neck-white-t-shirt-fraud-collection-logo-print-t-shirt": {
            "designer": "Fraud FD",
            "price": 40,
            "name": "Crew Neck White T-Shirt - Fraud Collection Logo Print",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/crew-neck-white-t-shirt-fraud-collection-logo-print-t-shirt",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164055/crew-neck-white-tshirt.jpg"
        },
        "orchid-patterned-bodycon-full-red-dress": {
            "designer": "Fraud FD",
            "price": 80,
            "name": "Orchid Patterned Bodycon Full Dress - Red",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/orchid-patterned-bodycon-full-red-dress",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164065/orchid-dress-red.jpg"
        },
        "short-sleave-t-shirt-white-with-red-decoration": {
            "designer": "Fraud FD",
            "price": 50,
            "name": "Short Sleeve T-Shirt White with Red Decoration",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/short-sleave-t-shirt-white-with-red-decoration",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164070/short-sleeve-white-tshirt.jpg"
        },
        "sweatshirt-white-fraud-signuture-canva": {
            "designer": "Fraud FD",
            "price": 65,
            "name": "Sweatshirt White - Fraud Signature Canvas",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/sweatshirt-white-fraud-signuture-canva",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164075/white-sweatshirt.jpg"
        },
        "khaki-t-shirt": {
            "designer": "Fraud FD",
            "price": 45,
            "name": "Khaki T-Shirt",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/khaki-t-shirt",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164080/khaki-tshirt.jpg"
        },
        "negro": {
            "designer": "Fraud FD",
            "price": 70,
            "name": "Negro Collection",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/negro",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164085/negro-collection.jpg"
        },
        "orchid-patterned-bodycon-full-dress": {
            "designer": "Fraud FD",
            "price": 85,
            "name": "Orchid Patterned Bodycon Full Dress",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/orchid-patterned-bodycon-full-dress",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164090/orchid-dress.jpg"
        },
        "lime-green-australian-cotton-track-suit": {
            "designer": "Fraud FD",
            "price": 95,
            "name": "Lime Green Australian Cotton Track Suit",
            "outfit_url": "https://www.fraudfd.com/collections/women/products/lime-green-australian-cotton-track-suit",
            "image_url": "https://res.cloudinary.com/dkzckcpzf/image/upload/v1750164095/lime-green-tracksuit.jpg"
        }
    }
    
    print("outfit_name", outfit_name)
    return outfit_data.get(outfit_name, {
        "name": "Unknown",
        "designer": "Unknown",
        "price": -1,
        "outfit_url": "",
        "image_url": ""
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)