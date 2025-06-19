import os
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np

# Path to SAM weights
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# Input and output directories
INPUT_ROOT = "sorted/omarmansoor"
OUTPUT_ROOT = "sam_segmented/sorted/omarmansoor"

# Create output root if it doesn't exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def get_all_image_paths(root_dir):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

def save_mask_and_crop(image_path, mask, output_mask_path, output_crop_path):
    # Save mask
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_mask_path, mask_img)
    # Save cropped image
    image = Image.open(image_path).convert("RGB")
    np_img = np.array(image)
    if len(mask.shape) == 2:
        mask_3c = np.stack([mask]*3, axis=-1)
    else:
        mask_3c = mask
    crop = np_img * mask_3c
    crop_img = Image.fromarray(crop)
    crop_img.save(output_crop_path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)

    image_paths = get_all_image_paths(INPUT_ROOT)
    print(f"Found {len(image_paths)} Omar Mansoor images to segment.")

    for img_path in image_paths:
        print(f"Segmenting: {img_path}")
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        # Use a single point in the center as prompt
        h, w, _ = image.shape
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        # Take the mask with the highest score
        best_mask = masks[np.argmax(scores)]
        # Prepare output paths
        rel_path = os.path.relpath(img_path, INPUT_ROOT)
        mask_path = os.path.join(OUTPUT_ROOT, rel_path.replace(".jpg", "_mask.png").replace(".jpeg", "_mask.png").replace(".png", "_mask.png"))
        crop_path = os.path.join(OUTPUT_ROOT, rel_path.replace(".jpg", "_crop.png").replace(".jpeg", "_crop.png").replace(".png", "_crop.png"))
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        save_mask_and_crop(img_path, best_mask, mask_path, crop_path)

if __name__ == "__main__":
    main() 