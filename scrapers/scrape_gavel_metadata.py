import json
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def get_all_product_links(base_url, total_pages=4):
    """
    Crawls the Gavel Paris shop pages to get all unique product links.
    """
    print("Setting up headless Chrome driver...")
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1200")
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
    except Exception as e:
        print(f"Error setting up webdriver: {e}")
        print("Please ensure Chrome is installed.")
        return []

    all_links = set()
    print(f"Scraping up to {total_pages} pages from {base_url}...")
    for page in range(1, total_pages + 1):
        url = f"{base_url}/page/{page}/"
        print(f"Accessing page: {url}")
        try:
            driver.get(url)
            time.sleep(3)  # Wait for dynamic content to load

            product_elements = driver.find_elements(
                By.CSS_SELECTOR, "h3.wd-entities-title a"
            )
            if not product_elements and page > 1:
                print(f"No products found on page {page}. Assuming it's the last page.")
                break

            for a in product_elements:
                href = a.get_attribute("href")
                if href:
                    all_links.add(href)
        except Exception as e:
            print(f"An error occurred while scraping {url}: {e}")
            break

    driver.quit()
    print(f"Found {len(all_links)} unique product links.")
    return list(all_links)


def create_metadata_file(product_links, output_file):
    """
    Creates a JSON metadata file from a list of product links.
    """
    metadata = {}
    print("Processing product links to create metadata...")
    for link in product_links:
        try:
            # Extract slug from URL
            product_slug = link.strip("/").split("/")[-1]
            # Convert slug to the key format used in the metadata file
            metadata_key = product_slug.replace("-", "_")

            if metadata_key:
                metadata[metadata_key] = {"checkout_link": link}
        except Exception as e:
            print(f"Could not process link {link}: {e}")

    print(f"Generated metadata for {len(metadata)} products.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the metadata to a JSON file
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata successfully saved to '{output_file}'")


if __name__ == "__main__":
    BASE_SHOP_URL = "https://gavelparis.com/shop"
    OUTPUT_FILE_PATH = "metadata/gavelparis-metadata.json"

    # Step 1: Scrape all product links
    links = get_all_product_links(BASE_SHOP_URL, total_pages=4)

    # Step 2: Create the metadata file
    if links:
        create_metadata_file(links, OUTPUT_FILE_PATH)
    else:
        print("No links were found, metadata file not created.") 