import os
import requests
from bs4 import BeautifulSoup
import re
import shutil
from urllib.parse import urljoin
from pathlib import Path
import time

def kebab_case(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

def download_image(url, dest_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(dest_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            print(f"✅ Downloaded: {dest_path}")
        else:
            print(f"❌ Failed to download {url} (status {response.status_code})")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")

def main():
    BASE_URL = "https://www.omarmansoor.com/all-collections"
    OUT_ROOT = Path("sorted/omarmansoor")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"🌐 Scraping: {BASE_URL}")
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all product blocks
    product_blocks = soup.find_all('div', class_=re.compile(r'product|collection|grid|item', re.I))
    if not product_blocks:
        # Fallback: look for Quick View blocks
        product_blocks = soup.find_all(text=re.compile(r'Quick View', re.I))

    # Find all product names and images
    found = 0
    for quick_view in soup.find_all(text=re.compile(r'Quick View', re.I)):
        parent = quick_view.find_parent('div')
        if not parent:
            continue
        # Get product name (next sibling or nearby text)
        name_tag = parent.find_next(string=re.compile(r'Dress|Jumpsuit|Maxi|Gown|Sleeves|Wrap|Bodice|Brooch|Collard|Hussar|A-Line|Empire|Embellished|Lace|Short|Cup|Coronet|Permian|Frankel|Barney|Tangerine|Lime|Neon|Ivory|Red|Bell|Blush|Peach|Blue|Frilled', re.I))
        if not name_tag:
            continue
        product_name = name_tag.strip()
        folder_name = kebab_case(product_name)
        out_dir = OUT_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Find image (look for img tag in parent or nearby)
        img_tag = parent.find('img')
        if not img_tag:
            img_tag = parent.find_next('img')
        if img_tag and img_tag.has_attr('src'):
            img_url = urljoin(BASE_URL, img_tag['src'])
            ext = os.path.splitext(img_url)[-1].split('?')[0]
            dest_path = out_dir / f"main{ext if ext else '.jpg'}"
            download_image(img_url, dest_path)
            found += 1
        else:
            print(f"❌ No image found for {product_name}")
        time.sleep(0.5)  # Be polite
    print(f"\n🎉 Done! Downloaded {found} products.")

if __name__ == "__main__":
    main() 