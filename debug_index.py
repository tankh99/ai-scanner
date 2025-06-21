import os
import pickle
from glob import glob

def debug_index_structure():
    """Debug the current index and folder structure"""
    
    # Check the images directory structure
    print("=== FOLDER STRUCTURE ANALYSIS ===")
    images_dir = "images/"
    
    if not os.path.exists(images_dir):
        print(f"ERROR: {images_dir} directory does not exist!")
        return
    
    designer_folders = []
    for item in os.listdir(images_dir):
        item_path = os.path.join(images_dir, item)
        if os.path.isdir(item_path):
            designer_folders.append(item)
            print(f"Designer folder found: {item}")
            
            # Count products in this designer folder
            product_count = 0
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    product_count += 1
                    # Count images in this product folder
                    image_count = len(glob(os.path.join(subitem_path, "*.jpg")) + 
                                    glob(os.path.join(subitem_path, "*.jpeg")) + 
                                    glob(os.path.join(subitem_path, "*.png")))
                    print(f"  - Product: {subitem} ({image_count} images)")
            
            print(f"  Total products for {item}: {product_count}")
    
    print(f"\nTotal designers found: {len(designer_folders)}")
    
    # Check if Omar Mansoor exists
    if "omarmansoor" in designer_folders:
        print("\n✓ Omar Mansoor folder found!")
    else:
        print("\n❌ Omar Mansoor folder NOT found!")
        print("Available designers:", designer_folders)
    
    # Check the index file
    print("\n=== INDEX FILE ANALYSIS ===")
    index_file = "reference_embeddings.pkl"
    
    if not os.path.exists(index_file):
        print(f"ERROR: {index_file} does not exist!")
        print("Run the indexing script first: python image_similarity_search.py --reindex")
        return
    
    try:
        with open(index_file, "rb") as f:
            indexed_data = pickle.load(f)
        
        print(f"Index contains {len(indexed_data)} products")
        
        # Analyze what's in the index
        indexed_designers = set()
        indexed_products = []
        
        for item in indexed_data:
            folder_path = item['product_folder']
            # Extract designer name from path
            path_parts = folder_path.split(os.sep)
            if len(path_parts) >= 3:
                designer = path_parts[-3]  # images/designer/product/
                product = path_parts[-1]
                indexed_designers.add(designer)
                indexed_products.append((designer, product))
        
        print(f"Indexed designers: {sorted(list(indexed_designers))}")
        
        # Check Omar Mansoor specifically
        omarmansoor_products = [p for d, p in indexed_products if d == "omarmansoor"]
        if omarmansoor_products:
            print(f"\n✓ Omar Mansoor products in index: {len(omarmansoor_products)}")
            print("Products:", omarmansoor_products[:5])  # Show first 5
        else:
            print("\n❌ No Omar Mansoor products found in index!")
            
        # Show some sample indexed products
        print(f"\nSample indexed products:")
        for i, (designer, product) in enumerate(indexed_products[:10]):
            print(f"  {i+1}. {designer}/{product}")
            
    except Exception as e:
        print(f"Error reading index file: {e}")

def check_specific_product(product_name):
    """Check if a specific product exists and has images"""
    print(f"\n=== CHECKING SPECIFIC PRODUCT: {product_name} ===")
    
    # Look for this product in the images directory
    found_paths = []
    for root, dirs, files in os.walk("images/"):
        for dir_name in dirs:
            if product_name.lower() in dir_name.lower():
                full_path = os.path.join(root, dir_name)
                found_paths.append(full_path)
                print(f"Found potential match: {full_path}")
                
                # Count images in this folder
                image_files = glob(os.path.join(full_path, "*.jpg")) + \
                             glob(os.path.join(full_path, "*.jpeg")) + \
                             glob(os.path.join(full_path, "*.png"))
                print(f"  Images found: {len(image_files)}")
                if image_files:
                    print(f"  Sample images: {[os.path.basename(f) for f in image_files[:3]]}")
    
    if not found_paths:
        print("No matching product folders found!")

if __name__ == "__main__":
    debug_index_structure()
    
    # You can also check specific products
    # check_specific_product("halter-neck-lace-crepe-dress") 