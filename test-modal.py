import modal
import os
import shutil
app = modal.App("example-get-started")

# Create image with embedded files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch", "torchvision", "ftfy", "regex", "clip",
        "pillow", "numpy", "fastapi", "python-multipart", "uvicorn"
    ])
    .add_local_file("reference_embeddings.pkl", "/models/reference_embeddings.pkl")
    .add_local_file("designer-metadata.json", "/models/designer-metadata.json")
)

volume = modal.Volume.from_name("metadata", create_if_missing=True)

@app.function(image=image, volumes={"/metadata": volume})
def copy_files_to_volume():
    """Copy embedded files to volume for persistence"""
    import shutil
    
    print("📤 Copying embedded files to Modal volume...")
    
    # Copy from embedded location to volume
    shutil.copy2("/models/reference_embeddings.pkl", "/metadata/reference_embeddings.pkl")
    shutil.copy2("/models/designer-metadata.json", "/metadata/designer-metadata.json")
    
    print("✅ Files copied to volume!")
    
    # List files in volume
    print("\n📁 Files in Modal volume:")
    for file in os.listdir("/metadata"):
        file_size = os.path.getsize(f"/metadata/{file}")
        print(f"   {file} ({file_size} bytes)")
    
    return True

@app.function(image=image, volumes={"/metadata": volume})
def read_files_from_volume():
    """Read files from Modal volume"""
    import json
    
    print("�� Reading files from Modal volume...")
    
    files = os.listdir("/metadata")
    print(f"Found {len(files)} files: {files}")
    
    if "designer-metadata.json" in files:
        with open("/metadata/designer-metadata.json", "r") as f:
            data = json.load(f)
            print(f"✅ Loaded JSON with {len(data)} items")
    
    return files

@app.function(image=image, volumes={"/metadata": volume})
@modal.fastapi_endpoint()
def classify_api(image: bytes, top_n: int = 5):
    """API endpoint for classification"""
    # Your classification logic here
    return {"success": True, "message": "Classification endpoint"}

@app.local_entrypoint()
def main():
    """Copy files to Modal volume."""
    print("�� Starting file copy to Modal...")
    copy_files_to_volume.remote()
    print("✅ Copy completed!")
    
    # Verify files were copied
    print("\n🔍 Verifying copied files...")
    files = read_files_from_volume.remote()
    print(f"✅ Found {len(files)} files in volume")