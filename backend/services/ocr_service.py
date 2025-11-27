"""
OCR Service for ProductLens AI.
Handles text extraction from images using Tesseract OCR.
"""

import io
from typing import Optional
import logging

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRService:
    """
    Service class for extracting text from images using Tesseract OCR.
    
    Includes preprocessing steps to improve OCR accuracy for handwritten text.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the OCRService.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional).
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract and Pillow are required. "
                "Install with: pip install pytesseract Pillow"
            )
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Verify tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.warning(f"Tesseract may not be installed: {e}")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image to improve OCR accuracy.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Preprocessed PIL Image.
        """
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small (helps with handwriting)
        min_dimension = 1000
        if min(image.size) < min_dimension:
            scale = min_dimension / min(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def extract_text(self, image_data: bytes) -> str:
        """
        Extract text from image bytes.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            Extracted text string.
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Configure Tesseract for handwriting
            custom_config = r"--oem 3 --psm 6"
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image, 
                config=custom_config
            )
            
            # Clean up the extracted text
            text = self._clean_extracted_text(text)
            
            logger.info(f"Extracted text: {text[:100]}..." if len(text) > 100 else f"Extracted text: {text}")
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from an image file.
        
        Args:
            file_path: Path to the image file.
            
        Returns:
            Extracted text string.
        """
        try:
            with open(file_path, "rb") as f:
                return self.extract_text(f.read())
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean up OCR-extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        text = " ".join(lines)
        
        # Remove multiple spaces
        import re
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
