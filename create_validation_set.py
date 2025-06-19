import os
import shutil
import random
from pathlib import Path

def move_files(files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        shutil.move(f, os.path.join(dest_dir, os.path.basename(f)))

def create_validation_set():
    random.seed(42)
    designers_root = Path('sam_segmented/sorted')
    val_root = Path('validation_set')
    val_root.mkdir(exist_ok=True)

    allowed_designers = {'omarmansoor', 'marinakarelyan', 'gavelparis', 'Reaumonde'}

    # 1. Omar Mansoor: select 5 random products for validation
    omar_dir = designers_root / 'omarmansoor'
    if omar_dir.exists():
        products = [p for p in omar_dir.iterdir() if p.is_dir()]
        val_products = random.sample(products, min(5, len(products)))
        for prod in val_products:
            dest = val_root / 'omarmansoor' / prod.name
            for img in prod.glob('*_crop.png'):
                move_files([str(img)], dest)
            for img in prod.glob('*_mask.png'):
                move_files([str(img)], dest)
        print(f"Omar Mansoor: {len(val_products)} products moved to validation set.")

    # 2. Other allowed designers: 20-30% of images per product
    for designer in designers_root.iterdir():
        if not designer.is_dir() or designer.name not in allowed_designers or designer.name == 'omarmansoor':
            continue
        for product in designer.iterdir():
            if not product.is_dir():
                continue
            images = list(product.glob('*_crop.png'))
            if len(images) < 2:
                continue
            val_count = max(1, int(0.2 * len(images)))
            val_images = random.sample(images, val_count)
            dest = val_root / designer.name / product.name
            move_files([str(img) for img in val_images], dest)
            for img in product.glob('*_mask.png'):
                mask_base = img.stem.replace('_mask', '')
                if any(mask_base in vimg.stem for vimg in val_images):
                    move_files([str(img)], dest)
    print("Validation set created at ./validation_set/ (only for attending designers)")

if __name__ == "__main__":
    create_validation_set() 