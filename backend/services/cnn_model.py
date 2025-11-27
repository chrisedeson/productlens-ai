"""
CNN Model Service for ProductLens AI.
Handles product image classification using a custom CNN model.
"""

import io
import os
from typing import Optional, Tuple, List
import logging
import numpy as np

try:
    from PIL import Image
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNModel:
    """
    Service class for CNN-based product image classification.
    
    Builds a CNN from scratch (no pre-trained models) to classify
    products into one of 10 categories.
    """
    
    # Product classes from CNN_Model_Train_Data.csv
    CLASSES = [
        "LUNCH BAG PINK POLKADOT",
        "ALARM CLOCK BAKELIKE RED",
        "CHOCOLATE HOT WATER BOTTLE",
        "SPOTTY BUNTING",
        "LUNCH BAG WOODLAND",
        "REX CASH+CARRY JUMBO SHOPPER",
        "JUMBO STORAGE BAG SUKI",
        "RETROSPOT TEA SET CERAMIC 11 PC",
        "6 RIBBONS RUSTIC CHARM",
        "REGENCY CAKESTAND 3 TIER"
    ]
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the CNNModel.
        
        Args:
            model_path: Path to saved model weights (optional).
            image_size: Target image size (height, width).
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            )
        
        self.image_size = image_size
        self.num_classes = len(self.CLASSES)
        self.model: Optional[keras.Model] = None
        self.model_path = model_path
        
        # Load existing model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No model loaded. Call build_model() to create a new model.")
    
    def build_model(self) -> keras.Model:
        """
        Build the CNN model architecture from scratch.
        
        Returns:
            Compiled Keras model.
        """
        logger.info("Building CNN model from scratch...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.image_size[0], self.image_size[1], 3)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation="softmax")
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.model = model
        logger.info("Model built successfully")
        model.summary()
        
        return model
    
    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        epochs: int = 50,
        callbacks: Optional[List] = None
    ) -> keras.callbacks.History:
        """
        Train the CNN model.
        
        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            epochs: Number of training epochs.
            callbacks: Optional list of Keras callbacks.
            
        Returns:
            Training history.
        """
        if self.model is None:
            self.build_model()
        
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        
        logger.info("Training complete!")
        return history
    
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Build or load a model first.")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a model from disk."""
        try:
            self.model = keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess an image for prediction.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            Preprocessed image as numpy array.
        """
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to target size
        image = image.resize(self.image_size)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_data: bytes) -> Tuple[str, float]:
        """
        Predict the product class for an image.
        
        Args:
            image_data: Raw image bytes.
            
        Returns:
            Tuple of (predicted class name, confidence score).
        """
        if self.model is None:
            logger.error("No model loaded for prediction")
            return "", 0.0
        
        try:
            # Preprocess the image
            img_array = self.preprocess_image(image_data)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get the predicted class
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            predicted_class = self.CLASSES[class_idx]
            
            logger.info(f"Prediction: {predicted_class} ({confidence:.2%})")
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "", 0.0
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return self.CLASSES.copy()
