"""
Image Scraper Service for ProductLens AI.
Handles scraping product images for CNN training data.
"""

import os
import time
import hashlib
import requests
from typing import List, Optional
import logging
from io import BytesIO

try:
    from PIL import Image
    from duckduckgo_search import DDGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageScraper:
    """
    Service class for scraping product images from the web.
    
    Uses DuckDuckGo Image Search (no API key required) to find
    and download product images for CNN training.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the ImageScraper.
        
        Args:
            output_dir: Directory to save downloaded images.
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow and duckduckgo-search are required. "
                "Install with: pip install Pillow duckduckgo-search"
            )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Request headers to avoid blocks
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def search_images(
        self, 
        query: str, 
        max_results: int = 100
    ) -> List[str]:
        """
        Search for images using DuckDuckGo.
        
        Args:
            query: Search query.
            max_results: Maximum number of image URLs to return.
            
        Returns:
            List of image URLs.
        """
        logger.info(f"Searching for images: {query}")
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    query,
                    max_results=max_results
                ))
            
            urls = [r["image"] for r in results if "image" in r]
            logger.info(f"Found {len(urls)} image URLs")
            return urls
            
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []
    
    def download_image(
        self, 
        url: str, 
        save_path: str,
        min_size: int = 100,
        max_size: int = 2000
    ) -> bool:
        """
        Download and validate an image.
        
        Args:
            url: Image URL.
            save_path: Path to save the image.
            min_size: Minimum dimension in pixels.
            max_size: Maximum dimension in pixels.
            
        Returns:
            True if download successful, False otherwise.
        """
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=10,
                stream=True
            )
            response.raise_for_status()
            
            # Load and validate image
            image = Image.open(BytesIO(response.content))
            
            # Check dimensions
            width, height = image.size
            if width < min_size or height < min_size:
                logger.debug(f"Image too small: {width}x{height}")
                return False
            
            # Resize if too large
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Save as JPEG
            image.save(save_path, "JPEG", quality=85)
            return True
            
        except Exception as e:
            logger.debug(f"Error downloading {url}: {e}")
            return False
    
    def scrape_product_images(
        self, 
        product_name: str,
        num_images: int = 100,
        search_variations: Optional[List[str]] = None
    ) -> int:
        """
        Scrape images for a specific product.
        
        Args:
            product_name: Name of the product.
            num_images: Target number of images.
            search_variations: Additional search terms to try.
            
        Returns:
            Number of images successfully downloaded.
        """
        # Create class directory
        class_name = self._sanitize_class_name(product_name)
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate search queries
        queries = [product_name]
        if search_variations:
            queries.extend([f"{product_name} {v}" for v in search_variations])
        else:
            # Default variations
            queries.extend([
                f"{product_name} product",
                f"{product_name} retail",
                f"buy {product_name}",
            ])
        
        downloaded = 0
        seen_hashes = set()
        
        for query in queries:
            if downloaded >= num_images:
                break
            
            urls = self.search_images(query, max_results=num_images)
            
            for url in urls:
                if downloaded >= num_images:
                    break
                
                # Generate unique filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
                filename = f"{class_name}_{downloaded:04d}_{url_hash}.jpg"
                save_path = os.path.join(class_dir, filename)
                
                # Skip if already exists
                if os.path.exists(save_path):
                    continue
                
                # Download and save
                if self.download_image(url, save_path):
                    # Check for duplicates using image hash
                    img_hash = self._get_image_hash(save_path)
                    if img_hash in seen_hashes:
                        os.remove(save_path)
                        continue
                    
                    seen_hashes.add(img_hash)
                    downloaded += 1
                    
                    if downloaded % 10 == 0:
                        logger.info(f"Downloaded {downloaded}/{num_images} images for {product_name}")
                
                # Rate limiting
                time.sleep(0.5)
        
        logger.info(f"Finished scraping {downloaded} images for {product_name}")
        return downloaded
    
    def scrape_all_products(
        self, 
        products: List[str],
        num_images_per_product: int = 100
    ) -> dict:
        """
        Scrape images for all products.
        
        Args:
            products: List of product names.
            num_images_per_product: Target images per product.
            
        Returns:
            Dictionary of product -> number of images downloaded.
        """
        results = {}
        
        for i, product in enumerate(products, 1):
            logger.info(f"Scraping product {i}/{len(products)}: {product}")
            count = self.scrape_product_images(product, num_images_per_product)
            results[product] = count
            
            # Pause between products
            time.sleep(2)
        
        return results
    
    def _sanitize_class_name(self, name: str) -> str:
        """Convert product name to valid directory name."""
        # Replace special characters with underscores
        sanitized = "".join(c if c.isalnum() else "_" for c in name)
        # Remove multiple underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_").lower()
    
    def _get_image_hash(self, path: str) -> str:
        """Generate a hash for an image to detect duplicates."""
        try:
            with Image.open(path) as img:
                # Resize to small size for hashing
                img = img.resize((8, 8)).convert("L")
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                bits = "".join("1" if p > avg else "0" for p in pixels)
                return bits
        except Exception:
            return ""
