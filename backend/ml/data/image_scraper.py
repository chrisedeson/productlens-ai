"""
Image Scraper for ProductLens AI.

Handles scraping product images from the web for CNN training.
"""

import os
import time
import hashlib
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
import logging

import requests

from core.base_service import BaseService
from core.exceptions import ValidationError, ExternalAPIError, ConfigurationError

# Optional dependencies
try:
    from PIL import Image
    from duckduckgo_search import DDGS
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    Image = None
    DDGS = None

logging.basicConfig(level=logging.INFO)


@dataclass
class ScrapingConfig:
    """
    Configuration for image scraping.
    
    Attributes:
        min_size: Minimum image dimension in pixels.
        max_size: Maximum image dimension (will resize if larger).
        quality: JPEG quality for saved images.
        rate_limit: Seconds between downloads.
        timeout: Request timeout in seconds.
        max_retries: Maximum download retry attempts.
    """
    min_size: int = 100
    max_size: int = 2000
    quality: int = 85
    rate_limit: float = 0.5
    timeout: int = 10
    max_retries: int = 3
    
    # Default search variations
    default_variations: List[str] = field(default_factory=lambda: [
        "product",
        "retail",
        "buy"
    ])
    
    # Request headers
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class ScrapingResult:
    """
    Results from an image scraping operation.
    
    Attributes:
        product_name: Name of the product scraped.
        images_downloaded: Number of images successfully downloaded.
        images_failed: Number of failed download attempts.
        duplicates_skipped: Number of duplicate images skipped.
        output_directory: Path where images were saved.
    """
    product_name: str
    images_downloaded: int
    images_failed: int
    duplicates_skipped: int
    output_directory: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "product_name": self.product_name,
            "images_downloaded": self.images_downloaded,
            "images_failed": self.images_failed,
            "duplicates_skipped": self.duplicates_skipped,
            "output_directory": self.output_directory,
            "success_rate": f"{self.images_downloaded / max(1, self.images_downloaded + self.images_failed) * 100:.1f}%"
        }


class ImageScraper(BaseService):
    """
    Service for scraping product images from the web.
    
    Uses DuckDuckGo Image Search (no API key required) to find
    and download product images for CNN training.
    
    Features:
    - Automatic duplicate detection using perceptual hashing
    - Image validation and resizing
    - Rate limiting to avoid blocks
    - Multiple search variations for better coverage
    
    Example:
        >>> scraper = ImageScraper(output_dir="./images")
        >>> result = scraper.scrape_product("red wine glass", num_images=50)
        >>> print(f"Downloaded {result.images_downloaded} images")
    """
    
    def __init__(
        self,
        output_dir: str,
        config: Optional[ScrapingConfig] = None
    ):
        """
        Initialize the ImageScraper.
        
        Args:
            output_dir: Base directory to save downloaded images.
            config: Scraping configuration options.
            
        Raises:
            ConfigurationError: If required dependencies are not installed.
        """
        super().__init__("ImageScraper")
        
        if not DEPENDENCIES_AVAILABLE:
            raise ConfigurationError(
                "dependencies",
                "Pillow and duckduckgo-search required. "
                "Install with: pip install Pillow duckduckgo-search"
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or ScrapingConfig()
        self._headers = {"User-Agent": self.config.user_agent}
        
        self._mark_initialized()
        self.logger.info(f"ImageScraper initialized, output: {output_dir}")
    
    def search_images(
        self,
        query: str,
        max_results: int = 100
    ) -> List[str]:
        """
        Search for images using DuckDuckGo.
        
        Args:
            query: Search query string.
            max_results: Maximum URLs to return.
            
        Returns:
            List of image URLs.
        """
        self.logger.info(f"Searching: '{query}' (max {max_results})")
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results))
            
            urls = [r["image"] for r in results if "image" in r]
            self.logger.info(f"Found {len(urls)} image URLs")
            return urls
            
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def download_image(
        self,
        url: str,
        save_path: str
    ) -> bool:
        """
        Download and validate an image.
        
        Args:
            url: Image URL.
            save_path: Path to save the image.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            response = requests.get(
                url,
                headers=self._headers,
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Load and validate
            image = Image.open(BytesIO(response.content))
            
            # Check minimum size
            width, height = image.size
            if width < self.config.min_size or height < self.config.min_size:
                self.logger.debug(f"Image too small: {width}x{height}")
                return False
            
            # Resize if too large
            if width > self.config.max_size or height > self.config.max_size:
                ratio = min(
                    self.config.max_size / width,
                    self.config.max_size / height
                )
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Save as JPEG
            image.save(save_path, "JPEG", quality=self.config.quality)
            return True
            
        except Exception as e:
            self.logger.debug(f"Download failed: {url} - {e}")
            return False
    
    def scrape_product(
        self,
        product_name: str,
        num_images: int = 100,
        search_variations: Optional[List[str]] = None
    ) -> ScrapingResult:
        """
        Scrape images for a specific product.
        
        Args:
            product_name: Name of the product.
            num_images: Target number of images.
            search_variations: Additional search term variations.
            
        Returns:
            ScrapingResult with download statistics.
        """
        # Create class directory
        class_name = self._sanitize_name(product_name)
        class_dir = self.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Build search queries
        variations = search_variations or self.config.default_variations
        queries = [product_name]
        queries.extend([f"{product_name} {v}" for v in variations])
        
        # Track statistics
        downloaded = 0
        failed = 0
        duplicates = 0
        seen_hashes: Set[str] = set()
        
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
                save_path = class_dir / filename
                
                # Skip existing
                if save_path.exists():
                    continue
                
                # Download
                if self.download_image(url, str(save_path)):
                    # Check for duplicates
                    img_hash = self._get_image_hash(str(save_path))
                    if img_hash in seen_hashes:
                        save_path.unlink()
                        duplicates += 1
                        continue
                    
                    seen_hashes.add(img_hash)
                    downloaded += 1
                    
                    if downloaded % 10 == 0:
                        self.logger.info(
                            f"Progress: {downloaded}/{num_images} for {product_name}"
                        )
                else:
                    failed += 1
                
                # Rate limiting
                time.sleep(self.config.rate_limit)
        
        self.logger.info(
            f"Completed: {downloaded} images for {product_name} "
            f"({failed} failed, {duplicates} duplicates)"
        )
        
        return ScrapingResult(
            product_name=product_name,
            images_downloaded=downloaded,
            images_failed=failed,
            duplicates_skipped=duplicates,
            output_directory=str(class_dir)
        )
    
    def scrape_all(
        self,
        products: List[str],
        num_images_per_product: int = 100
    ) -> Dict[str, ScrapingResult]:
        """
        Scrape images for multiple products.
        
        Args:
            products: List of product names.
            num_images_per_product: Target images per product.
            
        Returns:
            Dictionary mapping product names to results.
        """
        results = {}
        
        for i, product in enumerate(products, 1):
            self.logger.info(f"Scraping {i}/{len(products)}: {product}")
            
            result = self.scrape_product(product, num_images_per_product)
            results[product] = result
            
            # Pause between products
            time.sleep(2)
        
        return results
    
    def _sanitize_name(self, name: str) -> str:
        """
        Convert product name to valid directory name.
        
        Args:
            name: Product name.
            
        Returns:
            Sanitized name safe for filesystem.
        """
        sanitized = "".join(c if c.isalnum() else "_" for c in name)
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_").lower()
    
    def _get_image_hash(self, path: str) -> str:
        """
        Generate perceptual hash for duplicate detection.
        
        Args:
            path: Path to image file.
            
        Returns:
            Hash string for the image.
        """
        try:
            with Image.open(path) as img:
                # Resize to 8x8 grayscale
                img = img.resize((8, 8)).convert("L")
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                bits = "".join("1" if p > avg else "0" for p in pixels)
                return bits
        except Exception:
            return ""
    
    def health_check(self) -> dict:
        """Check scraper health."""
        base = super().health_check()
        base["output_dir"] = str(self.output_dir)
        base["output_dir_exists"] = self.output_dir.exists()
        base["dependencies_available"] = DEPENDENCIES_AVAILABLE
        return base
