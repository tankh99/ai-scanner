import os
import torch
import clip
from PIL import Image
import numpy as np
from glob import glob
import pickle
from tqdm import tqdm

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
REFERENCE_IMAGES_DIR = "images/"
INDEX_FILE = "reference_embeddings.pkl"
INPUT_IMAGE_PATH = "test.png"

# --- Model Loading ---
print(f"Loading CLIP model '{MODEL_NAME}' on device '{DEVICE}'...")
try:
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    print("Please make sure you have installed OpenAI's CLIP package: pip install git+https://github.com/openai/CLIP.git")
    exit()

# --- Core Functions ---

def get_image_embedding(image_path):
    """Generates a vector embedding for a single image."""
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model.encode_image(image)
            # Normalize the embedding for cosine similarity
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Warning: Could not process image {image_path}. Skipping. Error: {e}")
        return None

def index_images(root_dir, index_file):
    """
    Crawls directories to create embeddings for products.
    Handles both nested (designer/product/images) and flat (designer/individual_images) structures.
    For flat structures, each image is treated as its own product.
    """
    print(f"Starting product indexing for directory: {root_dir}")

    # Find all the product folders and individual images
    all_product_items = []
    for designer_folder in os.listdir(root_dir):
        designer_path = os.path.join(root_dir, designer_folder)
        if not os.path.isdir(designer_path):
            continue
            
        # Check if this designer uses nested structure (designer/product/images)
        has_nested_products = False
        for subitem in os.listdir(designer_path):
            subitem_path = os.path.join(designer_path, subitem)
            if os.path.isdir(subitem_path):
                # Check if this subdirectory contains images (nested structure)
                has_images = any(
                    os.path.isfile(os.path.join(subitem_path, f)) 
                    for f in os.listdir(subitem_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                )
                if has_images:
                    has_nested_products = True
                    all_product_items.append(("folder", subitem_path))
        
        # If no nested products found, treat each image as its own product (flat structure)
        if not has_nested_products:
            # Check if the designer folder contains images directly
            image_files = [
                f for f in os.listdir(designer_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if image_files:
                print(f"Found flat structure for designer: {designer_folder} with {len(image_files)} images")
                for image_file in image_files:
                    image_path = os.path.join(designer_path, image_file)
                    all_product_items.append(("image", image_path))

    if not all_product_items:
        print("Error: No products found in the specified directory structure.")
        return

    print(f"Found {len(all_product_items)} products to index.")

    embeddings_data = []
    for item_type, item_path in tqdm(all_product_items, desc="Indexing Products"):
        if item_type == "folder":
            # Handle nested structure: average all images in the folder
            extensions = ["*.jpg", "*.jpeg", "*.png"]
            image_paths_in_folder = []
            for ext in extensions:
                image_paths_in_folder.extend(glob(os.path.join(item_path, ext)))

            if not image_paths_in_folder:
                continue

            folder_embeddings = []
            for img_path in image_paths_in_folder:
                embedding = get_image_embedding(img_path)
                if embedding is not None:
                    folder_embeddings.append(embedding)
            
            if not folder_embeddings:
                continue

            # Average the embeddings for the current folder to create a 'meta-embedding'
            avg_embedding = np.mean(folder_embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding)
            
            embeddings_data.append({"product_folder": item_path, "embedding": avg_embedding})
            
        elif item_type == "image":
            # Handle flat structure: each image is its own product
            embedding = get_image_embedding(item_path)
            if embedding is not None:
                # Create a virtual folder path for the image
                # Format: images/designer/image_name (without extension)
                designer_name = os.path.basename(os.path.dirname(item_path))
                image_name = os.path.splitext(os.path.basename(item_path))[0]
                virtual_folder = os.path.join("images", designer_name, image_name)
                
                embeddings_data.append({"product_folder": virtual_folder, "embedding": embedding})

    print(f"Successfully indexed {len(embeddings_data)} products.")

    # Save the indexed data to a file
    with open(index_file, "wb") as f:
        pickle.dump(embeddings_data, f)
    print(f"Product index saved to '{index_file}'")

def find_most_similar(input_path, index_file, top_n=5):
    """
    Finds the top N most similar products from the index to a given input image.
    """
    # 1. Load the pre-computed product embeddings
    print(f"Loading product index from '{index_file}'...")
    if not os.path.exists(index_file):
        print(f"Error: Index file not found at '{index_file}'.")
        print("Please run the script with '--reindex' first to create the index.")
        return
        
    with open(index_file, "rb") as f:
        reference_data = pickle.load(f)

    reference_embeddings = np.vstack([item['embedding'] for item in reference_data])
    
    # 2. Generate the embedding for the input image
    print(f"Generating embedding for input image: {input_path}")
    input_embedding = get_image_embedding(input_path)
    if input_embedding is None:
        print("Could not process the input image.")
        return

    # 3. Compute cosine similarity against product embeddings
    similarities = np.dot(reference_embeddings, input_embedding.T).flatten()
    
    # 4. Find the top N best matches
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]
    
    print(f"\n--- Top {top_n} Most Similar Products ---")
    for i, index in enumerate(top_n_indices):
        match_folder = reference_data[index]['product_folder']
        match_score = similarities[index]
        print(f"#{i+1}: Product: '{match_folder}', Score: {match_score:.4f}")

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find similar images using CLIP embeddings.")
    parser.add_argument(
        '--reindex', 
        action='store_true', 
        help="If set, re-creates the image index from the reference directory."
    )
    parser.add_argument(
        '--input',
        type=str,
        default=INPUT_IMAGE_PATH,
        help="Path to the input image you want to find matches for."
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=5,
        help="Number of top similar images to display."
    )
    args = parser.parse_args()

    # If reindex is flagged or the index file doesn't exist, create it
    if args.reindex or not os.path.exists(INDEX_FILE):
        index_images(REFERENCE_IMAGES_DIR, INDEX_FILE)

    # Perform the search
    if os.path.exists(args.input):
        find_most_similar(args.input, INDEX_FILE, top_n=args.top_n)
    else:
        print(f"Error: Input file not found at '{args.input}'") 