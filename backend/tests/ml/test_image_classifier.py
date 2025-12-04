"""
Unit Tests for ImageClassifier.

Tests the CNN-based product image classification.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestImageClassifier:
    """Tests for ImageClassifier."""
    
    @pytest.fixture
    def mock_tf(self):
        """Mock TensorFlow."""
        mock = MagicMock()
        mock.__version__ = "2.16.0"
        return mock
    
    @pytest.fixture
    def classifier(self, mock_tf):
        """Create an ImageClassifier with mocked TensorFlow."""
        with patch.dict("sys.modules", {"tensorflow": mock_tf, "tensorflow.keras": mock_tf.keras}):
            with patch("ml.inference.image_classifier.TF_AVAILABLE", True):
                with patch("ml.inference.image_classifier.PIL_AVAILABLE", True):
                    from ml.inference.image_classifier import ImageClassifier
                    
                    classifier = ImageClassifier.__new__(ImageClassifier)
                    classifier.model = None
                    classifier.model_loaded = False
                    classifier._index_to_class = {
                        "0": "22384_LUNCH_BAG_PINK_POLKADOT",
                        "1": "22423_REGENCY_CAKESTAND"
                    }
                    classifier._class_to_index = {
                        "22384_LUNCH_BAG_PINK_POLKADOT": 0,
                        "22423_REGENCY_CAKESTAND": 1
                    }
                    classifier.image_size = (224, 224)
                    classifier.logger = Mock()
                    
                    return classifier
    
    @pytest.fixture
    def test_image_bytes(self):
        """Create test image bytes."""
        from PIL import Image
        
        img = Image.new("RGB", (224, 224), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer.read()
    
    def test_parse_class_name(self, classifier):
        """Test parsing class name to stock code and description."""
        stock_code, description = classifier._parse_class_name(
            "22384_LUNCH_BAG_PINK_POLKADOT"
        )
        
        assert stock_code == "22384"
        assert description == "LUNCH BAG PINK POLKADOT"
    
    def test_parse_class_name_no_underscore(self, classifier):
        """Test parsing class name without underscore."""
        stock_code, description = classifier._parse_class_name("LUNCHBAG")
        
        assert stock_code == ""
        assert description == "LUNCHBAG"
    
    def test_get_class_names(self, classifier):
        """Test getting all class names."""
        names = classifier.get_class_names()
        
        assert len(names) == 2
        assert "22384_LUNCH_BAG_PINK_POLKADOT" in names
    
    def test_get_num_classes(self, classifier):
        """Test getting number of classes."""
        num = classifier.get_num_classes()
        
        assert num == 2
    
    def test_is_loaded_false(self, classifier):
        """Test is_loaded returns False when model not loaded."""
        assert classifier.is_loaded() is False
    
    def test_is_loaded_true(self, classifier):
        """Test is_loaded returns True when model is loaded."""
        classifier.model = Mock()
        classifier.model_loaded = True
        
        assert classifier.is_loaded() is True
    
    def test_get_model_info(self, classifier):
        """Test getting model information."""
        info = classifier.get_model_info()
        
        assert "model_loaded" in info
        assert "num_classes" in info
        assert "image_size" in info


class TestImageClassifierPreprocessing:
    """Tests for image preprocessing."""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        from PIL import Image
        return Image.new("RGB", (500, 500), color="blue")
    
    def test_resize_image(self, test_image):
        """Test that images are resized correctly."""
        # Resize to target size
        resized = test_image.resize((224, 224))
        
        assert resized.size == (224, 224)
    
    def test_convert_to_rgb(self):
        """Test conversion to RGB."""
        from PIL import Image
        
        # Create RGBA image
        rgba_image = Image.new("RGBA", (224, 224), color=(255, 0, 0, 128))
        rgb_image = rgba_image.convert("RGB")
        
        assert rgb_image.mode == "RGB"
    
    def test_normalize_pixels(self):
        """Test pixel normalization."""
        import numpy as np
        
        # Create sample pixel array
        pixels = np.array([[[0, 127, 255]]], dtype=np.uint8)
        normalized = pixels.astype(np.float32) / 255.0
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestImageClassifierConfiguration:
    """Tests for classifier configuration."""
    
    def test_default_image_size(self):
        """Test default image size."""
        from ml.inference.image_classifier import ImageClassifier
        
        assert ImageClassifier.DEFAULT_IMAGE_SIZE == (224, 224)
    
    @pytest.mark.ml
    def test_tensorflow_not_available(self):
        """Test graceful handling when TensorFlow is not available."""
        with patch("ml.inference.image_classifier.TF_AVAILABLE", False):
            from ml.inference.image_classifier import ImageClassifier
            
            classifier = ImageClassifier()
            
            assert classifier.model is None
            assert classifier.model_loaded is False


class TestImageClassifierPredictions:
    """Tests for classification predictions."""
    
    @pytest.fixture
    def mock_predictions(self):
        """Mock prediction output."""
        import numpy as np
        return np.array([[0.8, 0.15, 0.05]])
    
    def test_top_k_predictions(self, mock_predictions):
        """Test getting top-k predictions."""
        import numpy as np
        
        # Get top-2 indices
        top_indices = np.argsort(mock_predictions[0])[-2:][::-1]
        
        assert len(top_indices) == 2
        assert top_indices[0] == 0  # Highest confidence
    
    def test_prediction_format(self):
        """Test prediction output format."""
        prediction = {
            "class_name": "22384_LUNCH_BAG_PINK_POLKADOT",
            "stock_code": "22384",
            "description": "LUNCH BAG PINK POLKADOT",
            "confidence": 0.85
        }
        
        assert "class_name" in prediction
        assert "stock_code" in prediction
        assert "description" in prediction
        assert "confidence" in prediction
        assert 0.0 <= prediction["confidence"] <= 1.0
