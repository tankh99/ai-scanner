import torchvision.transforms as T

from tqdm import tqdm

import wandb

from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
import torch

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

class FashionDatasetPairs(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load all images and their labels
        self.images = []
        self.labels = []
        self.label_to_idx = {}
        self.label_to_indices = {}
        
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                self.label_to_idx[label] = len(self.label_to_idx)
                self.label_to_indices[self.label_to_idx[label]] = []
                for img_name in os.listdir(label_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_idx = len(self.images)
                        self.images.append(os.path.join(label_path, img_name))
                        self.labels.append(self.label_to_idx[label])
                        self.label_to_indices[self.label_to_idx[label]].append(img_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get anchor image
        anchor_img = Image.open(self.images[idx]).convert('RGB')
        anchor_label = self.labels[idx]
        
        # Randomly decide if we want a positive or negative pair
        should_get_same_class = np.random.randint(0, 2)
        
        if should_get_same_class:
            # Get another image from the same class
            same_class_indices = self.label_to_indices[anchor_label]
            if len(same_class_indices) == 1:
                # If only one image in this class, use augmentation to create a positive pair
                img2 = anchor_img
            else:
                # Randomly select a different image from the same class
                other_indices = [i for i in same_class_indices if i != idx]
                idx2 = np.random.choice(other_indices)
                img2 = Image.open(self.images[idx2]).convert('RGB')
        else:
            # Get an image from a different class
            other_labels = [l for l in self.label_to_indices.keys() if l != anchor_label]
            if not other_labels:
                # If no other classes exist, use augmentation to create a negative pair
                img2 = anchor_img
                should_get_same_class = 1  # Force it to be a positive pair
            else:
                # Randomly select a different class
                other_label = np.random.choice(other_labels)
                # Randomly select an image from that class
                idx2 = np.random.choice(self.label_to_indices[other_label])
                img2 = Image.open(self.images[idx2]).convert('RGB')
    
        
        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            img2 = self.transform(img2)
        
        return anchor_img, img2, torch.FloatTensor([int(should_get_same_class)])
    
# Create a strong augmentation pipeline
augmentation_pipeline = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNet, self).__init__()
        
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
                
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
    
    def forward_one(self, x):
        """Passes one image through the backbone to get its embedding."""
        return self.backbone(x)

    def forward(self, x1, x2):
        """Passes a pair of images through the network."""
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        contrastive_loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1-label) * (torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        )
        return contrastive_loss
    
    
def train_siamese(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', continue_training=True, checkpoint_path="./best_siamese_model.pth"):
    wandb.init()
    best_val_loss = float('inf')
    
    if continue_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        
        print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss}")
    
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for img1, img2, label in train_pbar:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            test_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
            for img1, img2, label in test_pbar:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()
                test_pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        
        # Log metrics
        wandb.log({
            'train_loss': running_loss/len(train_loader),
            'val_loss': val_loss,
            'epoch': epoch
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_siamese_model.pth')
     
def train():
    device= "mps"
    model = SiameseNet()
    model.to(device)
    dataset = FashionDatasetPairs(root_dir="downloaded_images/")
    test_dataset = FashionDatasetPairs(root_dir="test_imgs/")
    train_loader = DataLoader(
        dataset, 
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True
    )
    loss = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_siamese(
        model, 
        train_loader, 
        val_loader, 
        criterion=loss, 
        optimizer=optimizer, 
        num_epochs=10,
        device=device,
        continue_training=False,
        checkpoint_path="./best_siamese_model.pth"
    )
    
       
def predict_with_embeddings(model, image, reference_embeddings, reference_labels, device='cuda'):
    """
    Predict the class of an image using the trained Siamese network.
    
    Args:
        model: Trained SiameseNet model
        image_path: Path to the image to classify
        reference_embeddings: Dictionary mapping class names to their reference embeddings
        reference_labels: List of class names in the same order as reference_embeddings
        device: Device to run inference on
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Load and preprocess image
    
    # Handle different input types
    if isinstance(image, str):
        # If path is provided, load the image
        image = Image.open(image).convert('RGB')
    
    if isinstance(image, Image.Image):
        # If PIL Image, transform it
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device)
    elif isinstance(image, torch.Tensor):
        # If tensor, ensure it has the right shape and is on the right device
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
    else:
        raise TypeError("Image must be either a path string, PIL Image, or torch.Tensor")
    
    # Get embedding for the input image
    with torch.no_grad():
        embedding = model.forward_one(image)
    
    # Calculate distances to all reference embeddings
    distances = []
    for ref_emb in reference_embeddings:
        dist = F.pairwise_distance(embedding, ref_emb.unsqueeze(0))
        distances.append(dist.item())
    
    # Get the closest match
    min_dist_idx = np.argmin(distances)
    predicted_class = reference_labels[min_dist_idx]
    confidence = 1.0 / (1.0 + distances[min_dist_idx])  # Convert distance to confidence
    
    return predicted_class[1], confidence

def predict(image):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load("./best_siamese_model.pth")

    model = SiameseNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # First, create your reference gallery
    reference_embeddings = []
    reference_labels = []

    # Load and preprocess your reference images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    reference_images = []
    base_dir = "./downloaded_images/"
    for outfit in os.listdir(base_dir):
        outfit_dir = os.path.join(base_dir, outfit)
        if not os.path.isdir(outfit_dir):
            continue
        
        for img in os.listdir(outfit_dir):
            img_path = os.path.join(outfit_dir, img)
            reference_images.append((
                img_path,
                outfit
            ))
    # reference_images
    # For each reference image
    for image_path, label in reference_images:
        img = Image.open(image_path).convert('RGB')
        image_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get embedding
        with torch.no_grad():
            embedding = model.forward_one(image_tensor)
        
        reference_embeddings.append(embedding)
        reference_labels.append(label)


    return predict_with_embeddings(
        model, 
        image,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_images,
        device=device
    )