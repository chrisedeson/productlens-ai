"""
OCR Service for ProductLens AI.

Handles text extraction from images using Tesseract OCR and OpenAI Vision.
Optimized for handwritten shopping list recognition.
"""

from __future__ import annotations

import io
import base64
import re
from typing import Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from core.base_service import BaseService
from core.exceptions import (
    ValidationError,
    ServiceError,
    ExternalAPIError,
    ConfigurationError
)

# Optional dependencies
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    Image = None

try:
    from openai import OpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

logging.basicConfig(level=logging.INFO)


@dataclass
class OCRResult:
    """
    Container for OCR extraction results.
    
    Attributes:
        text: Extracted text content.
        confidence: Confidence score (0-1) if available.
        method: OCR method used ('openai_vision' or 'tesseract').
        items: Parsed shopping list items if applicable.
    """
    text: str
    confidence: Optional[float]
    method: str
    items: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "method": self.method,
            "items": self.items
        }


class OCRService(BaseService):
    """
    Service for extracting text from images.
    
    Supports two OCR engines:
    1. OpenAI Vision (GPT-4o-mini) - Primary, best for handwriting
    2. Tesseract OCR - Fallback, good for printed text
    
    The service includes preprocessing steps to improve accuracy
    and can parse shopping list items from extracted text.
    
    Attributes:
        use_openai: Whether OpenAI Vision is available.
        use_tesseract: Whether Tesseract is available.
    
    Example:
        >>> service = OCRService(openai_api_key="...")
        >>> result = service.extract_text(image_bytes)
        >>> print(result.text)
        >>> print(result.items)  # Parsed list items
    """
    
    # Image preprocessing settings
    MIN_DIMENSION = 1000
    CONTRAST_FACTOR = 2.0
    
    # OpenAI Vision settings
    VISION_MODEL = "gpt-4o-mini"
    VISION_MAX_TOKENS = 500
    
    VISION_SYSTEM_PROMPT = """You are an OCR assistant. Extract all text from the image exactly as written. 
If the text is handwritten, transcribe it accurately. 
Return ONLY the extracted text, nothing else. 
Do not add any explanations or formatting."""
    
    # Tesseract settings
    TESSERACT_CONFIG = "--oem 3 --psm 6"
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        tesseract_cmd: Optional[str] = None
    ):
        """
        Initialize the OCR Service.
        
        Args:
            openai_api_key: OpenAI API key for Vision OCR (recommended).
            tesseract_cmd: Path to Tesseract executable (optional).
            
        Raises:
            ConfigurationError: If neither OCR engine is available.
        """
        super().__init__("OCRService")
        
        self._openai_client = None
        self.use_openai = False
        self.use_tesseract = False
        
        # Initialize OpenAI Vision
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self._openai_client = OpenAI(api_key=openai_api_key)
                self.use_openai = True
                self.logger.info("OpenAI Vision OCR initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI Vision: {e}")
        
        # Initialize Tesseract
        if TESSERACT_AVAILABLE:
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            try:
                version = pytesseract.get_tesseract_version()
                self.use_tesseract = True
                self.logger.info(f"Tesseract OCR initialized (v{version})")
            except Exception as e:
                self.logger.warning(f"Tesseract not available: {e}")
        
        # Validate at least one engine is available
        if not self.use_openai and not self.use_tesseract:
            raise ConfigurationError(
                "ocr_engine",
                "No OCR engine available. Install OpenAI or Tesseract."
            )
        
        self._mark_initialized()
    
    def extract_text(self, image_data: bytes) -> OCRResult:
        """
        Extract text from image bytes.
        
        Uses OpenAI Vision as primary method (best for handwriting),
        falls back to Tesseract if OpenAI is unavailable or fails.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            OCRResult with extracted text and metadata.
            
        Raises:
            ValidationError: If image data is empty or invalid.
        """
        if not image_data:
            raise ValidationError("image_data", "Image data cannot be empty")
        
        # Validate image format
        if not self._is_valid_image(image_data):
            raise ValidationError("image_data", "Invalid image format")
        
        # Try OpenAI Vision first
        if self.use_openai:
            result = self._extract_with_openai(image_data)
            if result.text and not self._is_garbage_text(result.text):
                self._log_operation("extract_text", {"method": "openai_vision"})
                return result
            self.logger.info("OpenAI Vision returned no valid text, trying Tesseract...")
        
        # Fall back to Tesseract
        if self.use_tesseract:
            result = self._extract_with_tesseract(image_data)
            self._log_operation("extract_text", {"method": "tesseract"})
            return result
        
        # No valid result
        return OCRResult(
            text="",
            confidence=0.0,
            method="none",
            items=[]
        )
    
    def extract_text_from_file(self, file_path: str) -> OCRResult:
        """
        Extract text from an image file.
        
        Args:
            file_path: Path to the image file.
            
        Returns:
            OCRResult with extracted text.
            
        Raises:
            ValidationError: If file cannot be read.
        """
        try:
            with open(file_path, "rb") as f:
                return self.extract_text(f.read())
        except FileNotFoundError:
            raise ValidationError("file_path", f"File not found: {file_path}")
        except IOError as e:
            raise ValidationError("file_path", f"Cannot read file: {e}")
    
    def _extract_with_openai(self, image_data: bytes) -> OCRResult:
        """
        Extract text using OpenAI Vision API.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            OCRResult with extracted text.
        """
        if not self._openai_client:
            return OCRResult(text="", confidence=0.0, method="openai_vision", items=[])
        
        try:
            # Encode image
            base64_image = base64.b64encode(image_data).decode("utf-8")
            image_format = self._detect_image_format(image_data)
            
            response = self._openai_client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": self.VISION_SYSTEM_PROMPT},
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
                max_tokens=self.VISION_MAX_TOKENS
            )
            
            text = response.choices[0].message.content.strip()
            text = self._clean_text(text)
            
            self.logger.info(f"OpenAI Vision extracted: {text[:100]}...")
            
            return OCRResult(
                text=text,
                confidence=0.9,  # OpenAI doesn't provide confidence
                method="openai_vision",
                items=self._parse_list_items(text)
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI Vision error: {e}")
            return OCRResult(text="", confidence=0.0, method="openai_vision", items=[])
    
    def _extract_with_tesseract(self, image_data: bytes) -> OCRResult:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            OCRResult with extracted text.
        """
        if not TESSERACT_AVAILABLE:
            return OCRResult(text="", confidence=0.0, method="tesseract", items=[])
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data))
            processed = self._preprocess_image(image)
            
            # Extract text with confidence data
            data = pytesseract.image_to_data(
                processed,
                config=self.TESSERACT_CONFIG,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [
                int(conf) for conf in data["conf"] 
                if conf != "-1" and str(conf).isdigit()
            ]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.0
            
            # Extract text
            text = pytesseract.image_to_string(
                processed,
                config=self.TESSERACT_CONFIG
            )
            text = self._clean_text(text)
            
            self.logger.info(f"Tesseract extracted (conf={avg_confidence:.2f}): {text[:100]}...")
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                method="tesseract",
                items=self._parse_list_items(text)
            )
            
        except Exception as e:
            self.logger.error(f"Tesseract error: {e}")
            return OCRResult(text="", confidence=0.0, method="tesseract", items=[])
    
    def _preprocess_image(self, image: "PILImage") -> "PILImage":
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Preprocessed image.
        """
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.CONTRAST_FACTOR)
        
        # Apply sharpening
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small
        if min(image.size) < self.MIN_DIMENSION:
            scale = self.MIN_DIMENSION / min(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _is_valid_image(self, image_data: bytes) -> bool:
        """
        Check if image data is a valid image.
        
        Args:
            image_data: Raw bytes to check.
            
        Returns:
            True if valid image format.
        """
        # Check magic bytes
        if image_data[:4] == b"\x89PNG":
            return True
        if image_data[:2] == b"\xff\xd8":  # JPEG
            return True
        if image_data[:4] == b"GIF8":
            return True
        if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return True
        return False
    
    def _detect_image_format(self, image_data: bytes) -> str:
        """
        Detect image format from bytes.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            Format string (jpeg, png, etc.).
        """
        if image_data[:4] == b"\x89PNG":
            return "png"
        if image_data[:2] == b"\xff\xd8":
            return "jpeg"
        if image_data[:4] == b"GIF8":
            return "gif"
        if image_data[:4] == b"RIFF":
            return "webp"
        return "jpeg"  # Default
    
    def _is_garbage_text(self, text: str) -> bool:
        """
        Check if extracted text appears to be garbage/noise.
        
        Args:
            text: Text to check.
            
        Returns:
            True if text appears to be garbage.
        """
        if not text or len(text.strip()) < 3:
            return True
        
        # Count character types
        special_chars = sum(1 for c in text if c in '|=-[];{}()<>~`\\@#$%^&*_+')
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        # High ratio of special characters = garbage
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
        
        # Very few alphabetic characters = garbage
        if alpha_chars < len(text) * 0.3:
            return True
        
        # Known garbage patterns
        garbage_patterns = [
            r'[|=\-]{3,}',
            r'[{}()\[\]]{2,}',
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up OCR-extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Remove extra whitespace and empty lines
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        text = " ".join(lines)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def _parse_list_items(self, text: str) -> List[str]:
        """
        Parse shopping list items from extracted text.
        
        Handles various list formats:
        - Numbered lists (1. item, 2. item)
        - Bulleted lists (- item, * item)
        - Line-separated items
        
        Args:
            text: Extracted text.
            
        Returns:
            List of parsed items.
        """
        if not text:
            return []
        
        items = []
        
        # Split by common delimiters
        lines = re.split(r'[\n,;]', text)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove list markers
            line = re.sub(r'^[\d]+[.\)]\s*', '', line)  # 1. or 1)
            line = re.sub(r'^[-*•◦]\s*', '', line)  # Bullets
            line = line.strip()
            
            if line and len(line) > 1:
                items.append(line)
        
        return items
    
    def health_check(self) -> dict:
        """
        Check OCR service health.
        
        Returns:
            Health status dictionary.
        """
        base_health = super().health_check()
        base_health["openai_vision_available"] = self.use_openai
        base_health["tesseract_available"] = self.use_tesseract
        
        if self.use_tesseract and TESSERACT_AVAILABLE:
            try:
                version = pytesseract.get_tesseract_version()
                base_health["tesseract_version"] = str(version)
            except Exception:
                pass
        
        return base_health
