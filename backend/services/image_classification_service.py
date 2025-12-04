"""
Image Classification Service for ProductLens AI.
Uses the custom CNN model trained from scratch for product classification.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import io

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassificationService:
    """
    Service for classifying product images using the trained CNN model.
    
    The model was trained from scratch (no pre-trained models) on 10 product classes.
    """
    
    def __init__(self, model_path: Optional[Path] = None, class_mapping_path: Optional[Path] = None):
        """
        Initialize the classification service.
        
        Args:
            model_path: Path to the trained Keras model file
            class_mapping_path: Path to the class mapping JSON file
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for image classification")
        
        # Default paths
        base_path = Path(__file__).parent.parent / "models"
        
        if model_path is None:
            # Try different model files
            for model_file in ["simple_cnn_model.keras", "product_classifier.keras", "best_cnn_model.keras"]:
                potential_path = base_path / model_file
                if potential_path.exists():
                    model_path = potential_path
                    break
        
        if class_mapping_path is None:
            class_mapping_path = base_path / "class_mapping.json"
        
        # Load model
        if model_path and model_path.exists():
            logger.info(f"Loading model from {model_path}")
            self.model = keras.models.load_model(str(model_path))
            self.model_loaded = True
        else:
            logger.warning(f"Model not found at {model_path}. Classification will return default predictions.")
            self.model = None
            self.model_loaded = False
        
        # Load class mapping
        if class_mapping_path.exists():
            with open(class_mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
            self.index_to_class = self.class_mapping.get('index_to_class', {})
            self.class_to_index = self.class_mapping.get('class_to_index', {})
            self.image_size = tuple(self.class_mapping.get('image_size', [224, 224]))
        else:
            logger.warning("Class mapping not found. Using default values.")
            self.index_to_class = {}
            self.class_to_index = {}
            self.image_size = (224, 224)
        
        logger.info(f"ImageClassificationService initialized. Model loaded: {self.model_loaded}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess an image for model inference.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def classify(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """
        Classify a product image.
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and confidence scores
        """
        if not self.model_loaded:
            # Return default prediction if model not loaded
            return [{
                'class_name': 'unknown',
                'stock_code': 'N/A',
                'confidence': 0.0,
                'description': 'Model not loaded'
            }]
        
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top-k indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.index_to_class.get(str(idx), f"class_{idx}")
            
            # Parse class name (format: "22384_LUNCH_BAG_PINK_POLKADOT")
            parts = class_name.split('_', 1)
            stock_code = parts[0] if len(parts) > 1 else 'N/A'
            description = parts[1].replace('_', ' ') if len(parts) > 1 else class_name
            
            results.append({
                'class_name': class_name,
                'stock_code': stock_code,
                'description': description,
                'confidence': float(predictions[idx])
            })
        
        return results
    
    def classify_from_bytes(self, image_bytes: bytes, top_k: int = 5) -> List[Dict]:
        """
        Classify an image from raw bytes.
        
        Args:
            image_bytes: Raw image bytes
            top_k: Number of top predictions to return
            
        Returns:
            List of classification results
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self.classify(image, top_k)
    
    def classify_from_file(self, file_path: Path, top_k: int = 5) -> List[Dict]:
        """
        Classify an image from a file path.
        
        Args:
            file_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            List of classification results
        """
        image = Image.open(file_path)
        return self.classify(image, top_k)
    
    def get_class_names(self) -> List[str]:
        """Get list of all class names."""
        return list(self.class_to_index.keys())
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_loaded': self.model_loaded,
            'num_classes': len(self.index_to_class),
            'image_size': self.image_size,
            'class_names': self.get_class_names()
        }
