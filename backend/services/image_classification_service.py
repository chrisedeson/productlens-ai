"""
Image Classification Service for ProductLens AI.
Uses the custom CNN model trained from scratch for product classification,
with OpenAI Vision as a fallback for better generalization.
"""

import json
import logging
import base64
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

# OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassificationService:
    """
    Service for classifying product images using the trained CNN model
    with OpenAI Vision as a fallback for better generalization.
    
    The CNN model was trained from scratch (no pre-trained models) on 10 product classes.
    OpenAI Vision provides broader product recognition capabilities.
    """
    
    # Confidence threshold below which we use OpenAI Vision
    CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self, model_path: Optional[Path] = None, class_mapping_path: Optional[Path] = None, 
                 openai_api_key: Optional[str] = None):
        """
        Initialize the classification service.
        
        Args:
            model_path: Path to the trained Keras model file
            class_mapping_path: Path to the class mapping JSON file
            openai_api_key: OpenAI API key for Vision fallback
        """
        # Initialize OpenAI client
        self.openai_client = None
        if openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI Vision initialized for image classification")
        
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
        self.model = None
        self.model_loaded = False
        if TF_AVAILABLE and model_path and model_path.exists():
            try:
                logger.info(f"Loading model from {model_path}")
                self.model = keras.models.load_model(str(model_path))
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load CNN model: {e}")
        else:
            logger.warning(f"CNN model not available. Using OpenAI Vision only.")
        
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
        
        logger.info(f"ImageClassificationService initialized. CNN loaded: {self.model_loaded}, OpenAI: {self.openai_client is not None}")
    
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
    
    def _classify_with_openai_vision(self, image_bytes: bytes) -> List[Dict]:
        """
        Classify a product image using OpenAI Vision API.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            List of classification results
        """
        if not self.openai_client:
            return []
        
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Determine image type
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format.lower() if image.format else "jpeg"
            if image_format == "jpg":
                image_format = "jpeg"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a product identification assistant for an e-commerce platform. 
Analyze the product image and identify what product it is.

Respond in this exact JSON format:
{
    "product_name": "Short descriptive name of the product",
    "category": "Product category (e.g., bag, kitchenware, decoration, clothing, electronics)",
    "description": "Brief description of the product with key features",
    "confidence": 0.95
}

Be specific about the product type. For example:
- "Lunch Bag with Woodland Pattern" not just "bag"
- "Ceramic Tea Set" not just "dishes"
- "3-Tier Cake Stand" not just "stand"

Only respond with the JSON, no other text."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify this product. What is it?"
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
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Clean up response if it has markdown code blocks
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                
                result = json.loads(result_text)
                
                logger.info(f"OpenAI Vision identified: {result.get('product_name')} (confidence: {result.get('confidence', 0.9)})")
                
                return [{
                    'class_name': result.get('product_name', 'Unknown Product'),
                    'stock_code': 'VISION',
                    'description': result.get('product_name', 'Unknown Product'),
                    'category': result.get('category', 'unknown'),
                    'full_description': result.get('description', ''),
                    'confidence': float(result.get('confidence', 0.9)),
                    'source': 'openai_vision'
                }]
                
            except json.JSONDecodeError:
                # If JSON parsing fails, extract product name from text
                logger.warning(f"Failed to parse OpenAI Vision response as JSON: {result_text}")
                return [{
                    'class_name': result_text[:100],
                    'stock_code': 'VISION',
                    'description': result_text[:100],
                    'confidence': 0.8,
                    'source': 'openai_vision'
                }]
                
        except Exception as e:
            logger.error(f"Error with OpenAI Vision classification: {e}")
            return []
    
    def classify(self, image: Image.Image, top_k: int = 5, image_bytes: Optional[bytes] = None) -> List[Dict]:
        """
        Classify a product image.
        
        Uses CNN model first, falls back to OpenAI Vision if:
        - CNN confidence is below threshold
        - CNN model is not loaded
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            image_bytes: Optional raw image bytes for OpenAI Vision
            
        Returns:
            List of dictionaries with class names and confidence scores
        """
        cnn_results = []
        
        # Try CNN model first if available
        if self.model_loaded:
            # Preprocess image
            img_array = self.preprocess_image(image)
            
            # Get predictions
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Get top-k indices
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            for idx in top_indices:
                class_name = self.index_to_class.get(str(idx), f"class_{idx}")
                
                # Parse class name (format: "22384_LUNCH_BAG_PINK_POLKADOT")
                parts = class_name.split('_', 1)
                stock_code = parts[0] if len(parts) > 1 else 'N/A'
                description = parts[1].replace('_', ' ') if len(parts) > 1 else class_name
                
                cnn_results.append({
                    'class_name': class_name,
                    'stock_code': stock_code,
                    'description': description,
                    'confidence': float(predictions[idx]),
                    'source': 'cnn'
                })
            
            # If top CNN confidence is high enough, return CNN results
            if cnn_results and cnn_results[0]['confidence'] >= self.CONFIDENCE_THRESHOLD:
                logger.info(f"CNN confident prediction: {cnn_results[0]['description']} ({cnn_results[0]['confidence']:.1%})")
                return cnn_results
            
            logger.info(f"CNN confidence low ({cnn_results[0]['confidence']:.1%}), trying OpenAI Vision...")
        
        # Use OpenAI Vision as fallback or primary
        if self.openai_client and image_bytes:
            vision_results = self._classify_with_openai_vision(image_bytes)
            if vision_results:
                # Combine with CNN results if available
                if cnn_results:
                    # Add CNN results as alternatives
                    vision_results.extend(cnn_results[:3])
                return vision_results
        
        # Return CNN results if available, even with low confidence
        if cnn_results:
            return cnn_results
        
        # No classification available
        return [{
            'class_name': 'unknown',
            'stock_code': 'N/A',
            'confidence': 0.0,
            'description': 'Could not classify product',
            'source': 'none'
        }]
    
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
        return self.classify(image, top_k, image_bytes=image_bytes)
    
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
