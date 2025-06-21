import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle
import json
import clip
from typing import List, Dict, Tuple

class EnsembleModel:
    def __init__(self):
        self.clip_model = None
        self.clip_preprocess = None
        self.resnet_model = None
        self.class_to_dress_key = None
        self.product_index = None
        self.designer_metadata = None
        
        # Load both models
        self.load_models()
    
    def load_models(self):
        """Load both CLIP and ResNet models"""
        print("Loading ensemble models...")
        
        # Load CLIP
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            self.clip_model.eval()
            print("✓ CLIP model loaded")
        except Exception as e:
            print(f"✗ CLIP loading failed: {e}")
        
        # Load ResNet
        try:
            if os.path.exists('class_to_dress_key.json'):
                with open('class_to_dress_key.json', 'r') as f:
                    self.class_to_dress_key = json.load(f)
            
            if os.path.exists('best_resnet50.pth'):
                self.resnet_model = models.resnet50(weights=None)
                self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, len(self.class_to_dress_key))
                self.resnet_model.load_state_dict(torch.load('best_resnet50.pth', map_location="cpu"))
                self.resnet_model.eval()
                print("✓ ResNet model loaded")
        except Exception as e:
            print(f"✗ ResNet loading failed: {e}")
        
        # Load CLIP data
        try:
            if os.path.exists("reference_embeddings.pkl"):
                with open("reference_embeddings.pkl", "rb") as f:
                    self.product_index = pickle.load(f)
            if os.path.exists("designer-metadata.json"):
                with open("designer-metadata.json", "r") as f:
                    self.designer_metadata = json.load(f)
            print("✓ CLIP reference data loaded")
        except Exception as e:
            print(f"✗ CLIP data loading failed: {e}")
    
    def get_clip_prediction(self, image: Image.Image, top_n: int = 3) -> List[Dict]:
        """Get CLIP similarity search predictions"""
        if not all([self.clip_model, self.clip_preprocess, self.product_index, self.designer_metadata]):
            return []
        
        try:
            processed_image = self.clip_preprocess(image).unsqueeze(0)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(processed_image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
            query_embedding = embedding.cpu().numpy()
            
            reference_embeddings = np.vstack([item['embedding'] for item in self.product_index])
            similarities = np.dot(reference_embeddings, query_embedding.T).flatten()
            top_n_positions = np.argsort(similarities)[-top_n:][::-1]
            
            results = []
            for position in top_n_positions:
                indexed_item = self.product_index[position]
                product_folder_path = indexed_item['product_folder']
                product_folder_name = os.path.basename(product_folder_path)
                metadata_key = product_folder_name.replace('-', '_')
                metadata = self.designer_metadata.get(metadata_key, {})
                
                results.append({
                    "product_name": product_folder_name,
                    "confidence": float(similarities[position]),
                    "outfit_url": metadata.get("checkout_link", "URL not found"),
                    "model": "CLIP"
                })
            
            return results
        except Exception as e:
            print(f"CLIP prediction error: {e}")
            return []
    
    def get_resnet_prediction(self, image: Image.Image) -> Dict:
        """Get ResNet classification prediction"""
        if not self.resnet_model or not self.class_to_dress_key:
            return {}
        
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.resnet_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.class_to_dress_key.get(str(predicted_idx.item()), "Unknown")
            
            # Look up the URL in designer_metadata
            outfit_url = "URL not found"
            if self.designer_metadata and predicted_class in self.designer_metadata:
                outfit_url = self.designer_metadata[predicted_class].get("checkout_link", "URL not found")
            
            return {
                "product_name": predicted_class,
                "confidence": float(confidence.item()),
                "outfit_url": outfit_url,
                "model": "ResNet"
            }
        except Exception as e:
            print(f"ResNet prediction error: {e}")
            return {}
    
    def ensemble_predict(self, image: Image.Image) -> Dict:
        """Combine both models for robust prediction"""
        results = {
            "primary_prediction": None,
            "backup_prediction": None,
            "ensemble_confidence": 0.0,
            "model_agreement": False,
            "fallback_used": False
        }
        
        # Get predictions from both models
        clip_results = self.get_clip_prediction(image, top_n=3)
        resnet_result = self.get_resnet_prediction(image)
        
        # If ResNet is available and confident, use it as primary
        if resnet_result and resnet_result.get("confidence", 0) > 0.7:
            results["primary_prediction"] = resnet_result
            results["ensemble_confidence"] = resnet_result["confidence"]
            
            # Use CLIP as backup if available
            if clip_results:
                results["backup_prediction"] = clip_results[0]
                
                # Check if models agree (similar predictions)
                resnet_name = self.normalize_label(resnet_result["product_name"])
                clip_name = self.normalize_label(clip_results[0]["product_name"])
                results["model_agreement"] = self.labels_match(resnet_name, clip_name)
        
        # If ResNet is not confident, use CLIP as primary
        elif clip_results:
            results["primary_prediction"] = clip_results[0]
            results["ensemble_confidence"] = clip_results[0]["confidence"]
            results["fallback_used"] = True
            
            # Use ResNet as backup if available
            if resnet_result:
                results["backup_prediction"] = resnet_result
        
        # If both models fail, return a generic response
        else:
            results["primary_prediction"] = {
                "product_name": "Fashion item detected",
                "confidence": 0.5,
                "outfit_url": "Please try again with a clearer image",
                "model": "Fallback"
            }
            results["ensemble_confidence"] = 0.5
            results["fallback_used"] = True
        
        return results
    
    def normalize_label(self, label: str) -> str:
        """Normalize label for comparison"""
        return label.replace('-', '_').lower().strip()
    
    def labels_match(self, pred_label: str, true_label: str) -> bool:
        """Check if labels match after normalization"""
        pred_norm = self.normalize_label(pred_label)
        true_norm = self.normalize_label(true_label)
        return (pred_norm == true_norm or 
                pred_norm in true_norm or 
                true_norm in pred_norm)

# Global ensemble model instance
ensemble_model = None

def get_ensemble_model():
    """Get or create the ensemble model instance"""
    global ensemble_model
    if ensemble_model is None:
        ensemble_model = EnsembleModel()
    return ensemble_model 