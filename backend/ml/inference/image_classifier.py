"""
Image Classifier for ProductLens AI.

Handles product image classification using a trained CNN model.
"""

from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np

from core.base_service import BaseService
from core.exceptions import (
    ValidationError,
    ModelError,
    ConfigurationError
)
from ml.config import get_config, HuggingFaceConfig

# Type checking imports
if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)


@dataclass
class ClassificationResult:
    """
    Result from image classification.
    
    Attributes:
        class_name: Raw class name from model.
        stock_code: Product stock code extracted from class name.
        description: Product description extracted from class name.
        confidence: Prediction confidence (0-1).
    """
    class_name: str
    stock_code: str
    description: str
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "class_name": self.class_name,
            "stock_code": self.stock_code,
            "description": self.description,
            "confidence": round(self.confidence, 4)
        }


class ImageClassifier(BaseService):
    """
    Service for classifying product images using a trained CNN.
    
    The model classifies products into categories based on their
    visual appearance. Supports loading models from:
    - Local filesystem
    - Hugging Face Hub
    
    Features:
    - Automatic model download from Hugging Face
    - Image preprocessing and validation
    - Top-K predictions with confidence scores
    - Graceful degradation when model unavailable
    
    Example:
        >>> classifier = ImageClassifier()
        >>> results = classifier.classify_from_bytes(image_data)
        >>> for r in results:
        ...     print(f"{r.description}: {r.confidence:.2%}")
    """
    
    DEFAULT_IMAGE_SIZE = (224, 224)
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        class_mapping_path: Optional[Path] = None,
        download_from_hf: bool = True,
        hf_config: Optional[HuggingFaceConfig] = None
    ):
        """
        Initialize the Image Classifier.
        
        Args:
            model_path: Path to local model file.
            class_mapping_path: Path to class mapping JSON.
            download_from_hf: Whether to download from Hugging Face if not found.
            hf_config: Hugging Face configuration for model download.
            
        Raises:
            ConfigurationError: If TensorFlow is not available.
        """
        super().__init__("ImageClassifier")
        
        # Check for TensorFlow
        if not TF_AVAILABLE:
            # Allow initialization but mark as not loaded
            self.logger.warning("TensorFlow not available - classifier disabled")
            self.model = None
            self.model_loaded = False
            self._index_to_class = {}
            self._class_to_index = {}
            self.image_size = self.DEFAULT_IMAGE_SIZE
            self._mark_initialized()
            return
        
        if not PIL_AVAILABLE:
            raise ConfigurationError("dependencies", "Pillow is required")
        
        self.model = None
        self.model_loaded = False
        self._index_to_class: Dict[str, str] = {}
        self._class_to_index: Dict[str, int] = {}
        self.image_size = self.DEFAULT_IMAGE_SIZE
        
        # Get default paths from config
        config = get_config()
        hf_config = hf_config or config.huggingface
        
        # Determine model path
        if model_path is None:
            model_path = self._find_local_model(config.paths.model_dir)
        
        if class_mapping_path is None:
            class_mapping_path = config.paths.model_dir / "class_mapping.json"
        
        # Try to load model
        if model_path and model_path.exists():
            self._load_model(model_path)
        elif download_from_hf and HF_AVAILABLE:
            self._download_and_load_from_hf(hf_config, config.paths.model_dir)
        else:
            self.logger.warning("No model found - classifier will return default predictions")
        
        # Load class mapping
        self._load_class_mapping(class_mapping_path)
        
        self._mark_initialized()
        self.logger.info(f"ImageClassifier initialized, model_loaded={self.model_loaded}")
    
    def _find_local_model(self, model_dir: Path) -> Optional[Path]:
        """
        Find a model file in the model directory.
        
        Args:
            model_dir: Directory to search.
            
        Returns:
            Path to model file or None.
        """
        model_files = [
            "product_classifier.keras",
            "simple_cnn_model.keras",
            "best_cnn_model.keras"
        ]
        
        for filename in model_files:
            path = model_dir / filename
            if path.exists():
                return path
        
        return None
    
    def _load_model(self, model_path: Path) -> None:
        """
        Load model from filesystem.
        
        Args:
            model_path: Path to model file.
        """
        try:
            self.logger.info(f"Loading model from {model_path}")
            self.model = keras.models.load_model(str(model_path))
            self.model_loaded = True
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
            self.model_loaded = False
    
    def _download_and_load_from_hf(
        self,
        hf_config: HuggingFaceConfig,
        model_dir: Path
    ) -> None:
        """
        Download model from Hugging Face Hub.
        
        Args:
            hf_config: Hugging Face configuration.
            model_dir: Directory to save the model.
        """
        try:
            self.logger.info(f"Downloading model from {hf_config.repo_id}")
            
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = hf_hub_download(
                repo_id=hf_config.repo_id,
                filename=hf_config.model_filename,
                token=hf_config.token,
                local_dir=str(model_dir),
                revision=hf_config.revision
            )
            
            self._load_model(Path(model_path))
            
        except Exception as e:
            self.logger.error(f"Failed to download from HF: {e}")
    
    def _load_class_mapping(self, path: Path) -> None:
        """
        Load class name mappings.
        
        Args:
            path: Path to class mapping JSON.
        """
        if not path.exists():
            self.logger.warning("Class mapping not found, using defaults")
            return
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            self._index_to_class = data.get("index_to_class", {})
            self._class_to_index = data.get("class_to_index", {})
            self.image_size = tuple(data.get("image_size", self.DEFAULT_IMAGE_SIZE))
            
            self.logger.info(f"Loaded {len(self._index_to_class)} class mappings")
            
        except Exception as e:
            self.logger.error(f"Failed to load class mapping: {e}")
    
    def preprocess_image(self, image: "PILImage") -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Preprocessed numpy array with batch dimension.
        """
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Normalize to [0, 1]
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    
    def classify(
        self,
        image: "PILImage",
        top_k: int = 5
    ) -> List[ClassificationResult]:
        """
        Classify a product image.
        
        Args:
            image: PIL Image object.
            top_k: Number of top predictions to return.
            
        Returns:
            List of classification results sorted by confidence.
        """
        if not self.model_loaded:
            return [ClassificationResult(
                class_name="unknown",
                stock_code="N/A",
                description="Model not loaded",
                confidence=0.0
            )]
        
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top-k indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self._index_to_class.get(str(idx), f"class_{idx}")
            stock_code, description = self._parse_class_name(class_name)
            
            results.append(ClassificationResult(
                class_name=class_name,
                stock_code=stock_code,
                description=description,
                confidence=float(predictions[idx])
            ))
        
        return results
    
    def classify_from_bytes(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ) -> List[ClassificationResult]:
        """
        Classify an image from raw bytes.
        
        Args:
            image_bytes: Raw image bytes.
            top_k: Number of top predictions.
            
        Returns:
            List of classification results.
            
        Raises:
            ValidationError: If image data is invalid.
        """
        if not image_bytes:
            raise ValidationError("image_bytes", "Image data cannot be empty")
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self.classify(image, top_k)
        except Exception as e:
            raise ValidationError("image_bytes", f"Invalid image data: {e}")
    
    def classify_from_file(
        self,
        file_path: Path,
        top_k: int = 5
    ) -> List[ClassificationResult]:
        """
        Classify an image from a file.
        
        Args:
            file_path: Path to image file.
            top_k: Number of top predictions.
            
        Returns:
            List of classification results.
            
        Raises:
            ValidationError: If file cannot be read.
        """
        path = Path(file_path)
        if not path.exists():
            raise ValidationError("file_path", f"File not found: {file_path}")
        
        try:
            image = Image.open(path)
            return self.classify(image, top_k)
        except Exception as e:
            raise ValidationError("file_path", f"Cannot read image: {e}")
    
    def _parse_class_name(self, class_name: str) -> tuple:
        """
        Parse class name into stock code and description.
        
        Class names follow format: "22384_LUNCH_BAG_PINK_POLKADOT"
        
        Args:
            class_name: Raw class name.
            
        Returns:
            Tuple of (stock_code, description).
        """
        parts = class_name.split("_", 1)
        if len(parts) > 1:
            stock_code = parts[0]
            description = parts[1].replace("_", " ")
        else:
            stock_code = "N/A"
            description = class_name
        
        return stock_code, description
    
    def get_class_names(self) -> List[str]:
        """Get list of all class names."""
        return list(self._class_to_index.keys())
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self._index_to_class)
    
    def health_check(self) -> dict:
        """Check classifier health."""
        base = super().health_check()
        base["model_loaded"] = self.model_loaded
        base["num_classes"] = self.get_num_classes()
        base["image_size"] = self.image_size
        base["tensorflow_available"] = TF_AVAILABLE
        base["huggingface_available"] = HF_AVAILABLE
        return base
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for inference.
        
        Returns:
            True if model is loaded, False otherwise.
        """
        return self.model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_loaded": self.model_loaded,
            "num_classes": self.get_num_classes(),
            "image_size": self.image_size,
            "class_names": self.get_class_names()[:10],  # First 10 for brevity
            "tensorflow_version": tf.__version__ if TF_AVAILABLE else None
        }
