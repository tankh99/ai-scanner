import json
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def get_all_product_links_from_omar(base_url):
    """
    Crawls Omar Mansoor's website to get all unique product links.
    Handles their "Load More" pagination system.
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
    print(f"Accessing: {base_url}")
    
    try:
        driver.get(base_url)
        time.sleep(5)  # Wait for initial page load
        
        # Click "Load More" button multiple times to load all products
        load_more_attempts = 0
        max_attempts = 10  # Prevent infinite loops
        
        while load_more_attempts < max_attempts:
            try:
                # Look for the "Load More" button
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More')]"))
                )
                
                print(f"Clicking 'Load More' button (attempt {load_more_attempts + 1})")
                driver.execute_script("arguments[0].click();", load_more_button)
                time.sleep(3)  # Wait for new products to load
                load_more_attempts += 1
                
            except Exception as e:
                print("No more 'Load More' button found or reached end of products")
                break
        
        # Now collect all product links from the fully loaded page
        print("Collecting all product links...")
        product_elements = driver.find_elements(
            By.CSS_SELECTOR, "a[href*='/product-page/']"
        )
        
        for element in product_elements:
            href = element.get_attribute("href")
            if href and href not in all_links:
                all_links.add(href)
                
    except Exception as e:
        print(f"An error occurred while scraping {base_url}: {e}")
    
    finally:
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
            # Extract product name from URL (e.g., 'halter-neck-lace-crepe-dress')
            # URL pattern: https://www.omarmansoor.com/product-page/halter-neck-lace-crepe-dress
            product_slug = link.strip("/").split("/")[-1]
            
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
    # Omar Mansoor's main collections page
    COLLECTION_URL = "https://www.omarmansoor.com/all-collections"
    OUTPUT_FILE_PATH = "metadata/omarmansoor-metadata.json"

    # Step 1: Scrape all product links
    links = get_all_product_links_from_omar(COLLECTION_URL)

    # Step 2: Create the metadata file
    if links:
        create_metadata_file(links, OUTPUT_FILE_PATH)
    else:
        print("No links were found, metadata file not created.") 