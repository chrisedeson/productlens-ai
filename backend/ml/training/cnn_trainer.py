"""
CNN Trainer for ProductLens AI.

Handles training of the product classification CNN model.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging

from core.base_service import BaseService
from core.exceptions import ValidationError, ConfigurationError, ModelError
from ml.config import (
    get_config,
    MLConfig,
    CNNModelConfig,
    TrainingConfig,
    DataPaths
)

# Type checking imports
if TYPE_CHECKING:
    from tensorflow.keras import Model as KerasModel

# Optional dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    regularizers = None

logging.basicConfig(level=logging.INFO)


@dataclass
class TrainingResult:
    """
    Results from model training.
    
    Attributes:
        model_path: Path to saved model.
        epochs_trained: Number of epochs completed.
        final_accuracy: Final training accuracy.
        final_val_accuracy: Final validation accuracy.
        final_loss: Final training loss.
        final_val_loss: Final validation loss.
        history: Full training history.
    """
    model_path: str
    epochs_trained: int
    final_accuracy: float
    final_val_accuracy: float
    final_loss: float
    final_val_loss: float
    history: Dict[str, List[float]]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "epochs_trained": self.epochs_trained,
            "final_accuracy": round(self.final_accuracy, 4),
            "final_val_accuracy": round(self.final_val_accuracy, 4),
            "final_loss": round(self.final_loss, 4),
            "final_val_loss": round(self.final_val_loss, 4)
        }


class CNNTrainer(BaseService):
    """
    Trainer for the product classification CNN.
    
    Handles:
    - Model architecture creation
    - Training with callbacks
    - Model saving and checkpointing
    - Training history logging
    
    The trainer supports multiple base models:
    - MobileNetV2 (default, lightweight)
    - ResNet50
    - VGG16
    - InceptionV3
    - EfficientNetB0
    
    Example:
        >>> trainer = CNNTrainer()
        >>> result = trainer.train(
        ...     train_data=train_gen,
        ...     val_data=val_gen,
        ...     num_classes=10
        ... )
        >>> print(f"Final accuracy: {result.final_val_accuracy:.2%}")
    """
    
    def __init__(
        self,
        config: Optional[MLConfig] = None
    ):
        """
        Initialize the CNN trainer.
        
        Args:
            config: ML configuration.
            
        Raises:
            ConfigurationError: If TensorFlow is not available.
        """
        super().__init__("CNNTrainer")
        
        if not TF_AVAILABLE:
            raise ConfigurationError(
                "tensorflow",
                "TensorFlow is required for training"
            )
        
        self.config = config or get_config()
        self.model = None
        
        self._mark_initialized()
    
    def build_model(
        self,
        num_classes: int,
        model_config: Optional[CNNModelConfig] = None
    ) -> "KerasModel":
        """
        Build the CNN model architecture.
        
        Args:
            num_classes: Number of output classes.
            model_config: Model configuration.
            
        Returns:
            Compiled Keras model.
        """
        model_config = model_config or self.config.model
        input_shape = self.config.image.input_shape
        
        self.logger.info(
            f"Building {model_config.base_model} model "
            f"with {num_classes} classes"
        )
        
        # Get base model
        base_model = self._get_base_model(
            model_config.base_model,
            input_shape
        )
        
        if model_config.freeze_base:
            base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Add dense layers
        for units in model_config.dense_units:
            x = layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(model_config.l2_regularization)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(model_config.dropout_rate)(x)
        
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.training.learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.logger.info(f"Model built: {self.model.count_params():,} parameters")
        
        return self.model
    
    def _get_base_model(
        self,
        model_name: str,
        input_shape: tuple
    ) -> "KerasModel":
        """
        Get a pretrained base model.
        
        Args:
            model_name: Name of the base model.
            input_shape: Input tensor shape.
            
        Returns:
            Base model without top layers.
        """
        models = {
            "MobileNetV2": keras.applications.MobileNetV2,
            "ResNet50": keras.applications.ResNet50,
            "VGG16": keras.applications.VGG16,
            "InceptionV3": keras.applications.InceptionV3,
            "EfficientNetB0": keras.applications.EfficientNetB0
        }
        
        if model_name not in models:
            raise ValidationError(
                "base_model",
                f"Unknown model: {model_name}. Available: {list(models.keys())}"
            )
        
        return models[model_name](
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
    
    def build_simple_cnn(
        self,
        num_classes: int,
        input_shape: Optional[tuple] = None
    ) -> "KerasModel":
        """
        Build a simple CNN without transfer learning.
        
        This is useful for smaller datasets or when pretrained
        models are not suitable.
        
        Args:
            num_classes: Number of output classes.
            input_shape: Input tensor shape.
            
        Returns:
            Compiled Keras model.
        """
        input_shape = input_shape or self.config.image.input_shape
        
        self.logger.info(f"Building simple CNN with {num_classes} classes")
        
        self.model = keras.Sequential([
            keras.Input(shape=input_shape),
            
            # Block 1
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Block 2
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Block 3
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Block 4
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax")
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.training.learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.logger.info(f"Simple CNN built: {self.model.count_params():,} parameters")
        
        return self.model
    
    def train(
        self,
        train_data,
        val_data,
        num_classes: Optional[int] = None,
        epochs: Optional[int] = None,
        model_name: str = "product_classifier"
    ) -> TrainingResult:
        """
        Train the model.
        
        Args:
            train_data: Training data generator or dataset.
            val_data: Validation data generator or dataset.
            num_classes: Number of classes (auto-detected if not provided).
            epochs: Number of epochs (uses config default if not provided).
            model_name: Name for saved model file.
            
        Returns:
            TrainingResult with training metrics.
        """
        epochs = epochs or self.config.training.epochs
        
        # Build model if not already built
        if self.model is None:
            if num_classes is None:
                # Try to detect from data
                num_classes = getattr(train_data, "num_classes", 10)
            self.build_model(num_classes)
        
        # Create callbacks
        callbacks = self._create_callbacks(model_name)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Train
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model_path = self._save_model(model_name)
        
        # Create result
        result = TrainingResult(
            model_path=str(model_path),
            epochs_trained=len(history.history["loss"]),
            final_accuracy=history.history["accuracy"][-1],
            final_val_accuracy=history.history["val_accuracy"][-1],
            final_loss=history.history["loss"][-1],
            final_val_loss=history.history["val_loss"][-1],
            history=dict(history.history)
        )
        
        self.logger.info(
            f"Training complete: "
            f"acc={result.final_accuracy:.4f}, "
            f"val_acc={result.final_val_accuracy:.4f}"
        )
        
        return result
    
    def _create_callbacks(self, model_name: str) -> list:
        """
        Create training callbacks.
        
        Args:
            model_name: Name for checkpoint files.
            
        Returns:
            List of Keras callbacks.
        """
        training_config = self.config.training
        paths = self.config.paths
        
        # Ensure directories exist
        paths.ensure_directories()
        
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=training_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=training_config.reduce_lr_factor,
                patience=training_config.reduce_lr_patience,
                min_lr=training_config.min_lr,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=str(paths.checkpoints_dir / f"{model_name}_best.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=str(paths.logs_dir / f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}")
            )
        ]
        
        return callbacks
    
    def _save_model(self, model_name: str) -> Path:
        """
        Save the trained model.
        
        Args:
            model_name: Name for the model file.
            
        Returns:
            Path to saved model.
        """
        model_path = self.config.paths.model_dir / f"{model_name}.keras"
        self.model.save(str(model_path))
        self.logger.info(f"Model saved to {model_path}")
        return model_path
    
    def save_class_mapping(
        self,
        class_indices: Dict[str, int],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save class name to index mapping.
        
        Args:
            class_indices: Dictionary mapping class names to indices.
            output_path: Path for output file.
            
        Returns:
            Path to saved mapping file.
        """
        output_path = output_path or (self.config.paths.model_dir / "class_mapping.json")
        
        mapping = {
            "class_to_index": class_indices,
            "index_to_class": {str(v): k for k, v in class_indices.items()},
            "image_size": list(self.config.image.target_size)
        }
        
        with open(output_path, "w") as f:
            json.dump(mapping, f, indent=2)
        
        self.logger.info(f"Class mapping saved to {output_path}")
        return output_path
    
    def fine_tune(
        self,
        train_data,
        val_data,
        unfreeze_layers: int = 50,
        learning_rate: float = 1e-5,
        epochs: int = 10
    ) -> TrainingResult:
        """
        Fine-tune the model by unfreezing base layers.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            unfreeze_layers: Number of layers to unfreeze from the end.
            learning_rate: Learning rate for fine-tuning.
            epochs: Number of fine-tuning epochs.
            
        Returns:
            TrainingResult from fine-tuning.
        """
        if self.model is None:
            raise ModelError(
                "fine_tune",
                "No model to fine-tune. Train first."
            )
        
        self.logger.info(f"Fine-tuning: unfreezing last {unfreeze_layers} layers")
        
        # Unfreeze layers
        for layer in self.model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.train(
            train_data,
            val_data,
            epochs=epochs,
            model_name="product_classifier_finetuned"
        )
    
    def health_check(self) -> dict:
        """Check trainer health."""
        base = super().health_check()
        base["tensorflow_available"] = TF_AVAILABLE
        base["tensorflow_version"] = tf.__version__ if TF_AVAILABLE else None
        base["model_built"] = self.model is not None
        if self.model:
            base["model_params"] = self.model.count_params()
        return base
