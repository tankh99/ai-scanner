import json
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin


def get_all_product_links_from_shopify(collection_url):
    """
    Crawls a Shopify collection page to get all unique product links.
    It handles pagination by appending '?page=NUMBER'.
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
    page = 1
    while True:
        # Shopify pagination is typically handled with a '?page=' query parameter
        url_to_scrape = f"{collection_url}?page={page}"
        print(f"Accessing page: {url_to_scrape}")
        try:
            driver.get(url_to_scrape)
            time.sleep(4)  # Wait for dynamic content to load

            # This selector targets links within product cards. It's a common pattern
            # for Shopify themes but might need adjustment for custom themes.
            # We look for links inside an element with an id containing 'product-grid'.
            product_elements = driver.find_elements(
                By.CSS_SELECTOR, "div[id*='product-grid'] a"
            )

            if not product_elements:
                print(f"No products found on page {page}. Assuming it's the last page.")
                break

            found_new_links_on_page = False
            for a in product_elements:
                href = a.get_attribute("href")
                if href and href not in all_links:
                    all_links.add(href)
                    found_new_links_on_page = True
            
            # If no new links are found on the current page, we can stop.
            if not found_new_links_on_page and page > 1:
                print("No new products found, stopping pagination.")
                break

        except Exception as e:
            print(f"An error occurred while scraping {url_to_scrape}: {e}")
            break
        page += 1

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
            # Extract slug from URL (e.g., 'black-velvet-gown')
            product_slug = link.strip("/").split("/")[-1].split("?")[0]
            # Convert slug to the key format used in other metadata files
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
    # The main collection page for all products on many Shopify sites
    COLLECTION_URL = "https://marinakarelian.com/collections/all"
    OUTPUT_FILE_PATH = "metadata/marinakarelian-metadata.json"

    # Step 1: Scrape all product links
    links = get_all_product_links_from_shopify(COLLECTION_URL)

    # Step 2: Create the metadata file
    if links:
        create_metadata_file(links, OUTPUT_FILE_PATH)
    else:
        print("No links were found, metadata file not created.") 