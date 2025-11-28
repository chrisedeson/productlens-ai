"""
OCR Service for ProductLens AI.
Handles text extraction from images using Tesseract OCR and OpenAI Vision.
"""

import io
import base64
import re
from typing import Optional
import logging

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRService:
    """
    Service class for extracting text from images using Tesseract OCR
    with OpenAI Vision as a fallback for handwritten text.
    
    Includes preprocessing steps to improve OCR accuracy for handwritten text.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize the OCRService.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional).
            openai_api_key: OpenAI API key for vision-based OCR (optional).
        """
        self.openai_client = None
        
        # Initialize OpenAI client if available
        if openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI Vision OCR initialized successfully")
        
        # Initialize Tesseract as fallback
        if TESSERACT_AVAILABLE:
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR initialized successfully")
            except Exception as e:
                logger.warning(f"Tesseract may not be installed: {e}")
        
        if not self.openai_client and not TESSERACT_AVAILABLE:
            raise ImportError(
                "Either OpenAI or pytesseract/Pillow is required. "
                "Install with: pip install openai or pip install pytesseract Pillow"
            )
    
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
    
    def _extract_with_openai_vision(self, image_data: bytes) -> str:
        """
        Extract text from image using OpenAI Vision API.
        
        This is especially effective for handwritten text.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            Extracted text string.
        """
        if not self.openai_client:
            return ""
        
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            
            # Determine image type
            image = Image.open(io.BytesIO(image_data))
            image_format = image.format.lower() if image.format else "jpeg"
            if image_format == "jpg":
                image_format = "jpeg"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR assistant. Extract all text from the image exactly as written. "
                                   "If the text is handwritten, transcribe it accurately. "
                                   "Return ONLY the extracted text, nothing else. "
                                   "Do not add any explanations or formatting."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the text content."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI Vision extracted: {text[:100]}..." if len(text) > 100 else f"OpenAI Vision extracted: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Error with OpenAI Vision OCR: {e}")
            return ""
    
    def _is_garbage_text(self, text: str) -> bool:
        """
        Check if the extracted text appears to be garbage/noise.
        
        Args:
            text: Extracted text to check.
            
        Returns:
            True if text appears to be garbage.
        """
        if not text or len(text.strip()) < 3:
            return True
        
        # Count special characters vs alphanumeric
        special_chars = sum(1 for c in text if c in '|=-[];{}()<>~`\\@#$%^&*_+')
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        # If more than 30% special characters, likely garbage
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
        
        # If very few alphabetic characters
        if alpha_chars < len(text) * 0.3:
            return True
        
        # Check for common garbage patterns
        garbage_patterns = [
            r'[|=\-]{3,}',  # Multiple pipes, equals, dashes
            r'[{}()\[\]]{2,}',  # Multiple brackets
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def extract_text(self, image_data: bytes) -> str:
        """
        Extract text from image bytes.
        
        Uses OpenAI Vision as primary method (best for handwriting),
        falls back to Tesseract if OpenAI is unavailable.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            Extracted text string.
        """
        # Try OpenAI Vision first (best for handwritten text)
        if self.openai_client:
            text = self._extract_with_openai_vision(image_data)
            if text and not self._is_garbage_text(text):
                return self._clean_extracted_text(text)
            logger.info("OpenAI Vision returned no valid text, trying Tesseract...")
        
        # Fall back to Tesseract
        if TESSERACT_AVAILABLE:
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
                
                logger.info(f"Tesseract extracted: {text[:100]}..." if len(text) > 100 else f"Tesseract extracted: {text}")
                
                return text
                
            except Exception as e:
                logger.error(f"Error extracting text with Tesseract: {e}")
        
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
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
