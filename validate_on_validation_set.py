#!/usr/bin/env python3
"""
Validation script: Test all images in validation_set/ using the /classify endpoint.
Reports accuracy, confidence, and misclassifications.
"""
import requests
import time
from pathlib import Path

def test_image_classification(image_path, server_url="http://localhost:8000"):
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{server_url}/classify", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def get_validation_images():
    val_dir = Path("validation_set")
    images = []
    for designer_dir in val_dir.iterdir():
        if designer_dir.is_dir():
            for product_dir in designer_dir.iterdir():
                if product_dir.is_dir():
                    for img_file in product_dir.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg'] and '_crop' in img_file.stem:
                            images.append({
                                'path': str(img_file),
                                'designer': designer_dir.name,
                                'product': product_dir.name,
                                'filename': img_file.name
                            })
    return images

def main():
    print("🧪 Validating on Unseen Dresses (Validation Set)")
    print("=" * 60)
    val_images = get_validation_images()
    print(f"Found {len(val_images)} validation images.")
    results = []
    total_time = 0
    correct = 0
    for i, img in enumerate(val_images, 1):
        print(f"[{i}/{len(val_images)}] {img['designer']}/{img['product']}/{img['filename']}")
        start = time.time()
        result = test_image_classification(img['path'])
        elapsed = time.time() - start
        total_time += elapsed
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            results.append({**img, 'result': result, 'correct': False, 'confidence': 0, 'time': elapsed})
            continue
        pred_label = result['outfit_name']
        pred_designer, pred_product = pred_label.split('/', 1) if '/' in pred_label else (pred_label, '')
        is_correct = (pred_designer.lower() == img['designer'].lower() and pred_product.lower() == img['product'].lower())
        if is_correct:
            correct += 1
        print(f"   Predicted: {pred_label} | Confidence: {result['confidence']*100:.1f}% | {'✅' if is_correct else '❌'}")
        results.append({**img, 'result': result, 'correct': is_correct, 'confidence': result['confidence'], 'time': elapsed})
    print("\n" + "=" * 60)
    print(f"Validation Accuracy: {correct}/{len(val_images)} ({(correct/len(val_images))*100:.1f}%)")
    if results:
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        print(f"Average Confidence: {avg_conf*100:.1f}%")
        print(f"Average Time per Image: {total_time/len(results):.2f}s")
    print("\nMisclassifications:")
    for r in results:
        if not r['correct']:
            print(f" - {r['designer']}/{r['product']}/{r['filename']} -> {r['result'].get('outfit_name', 'ERROR')} (Conf: {r['confidence']*100:.1f}%)")
    print("\n🎉 Validation complete!")

if __name__ == "__main__":
    main() 