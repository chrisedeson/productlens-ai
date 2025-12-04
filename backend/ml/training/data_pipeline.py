"""
Data Pipeline for ProductLens AI CNN Training.

Handles data loading, augmentation, and batching.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from core.base_service import BaseService
from core.exceptions import ValidationError, ConfigurationError, DataError
from ml.config import ImageConfig, TrainingConfig, get_config

# Optional dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None

logging.basicConfig(level=logging.INFO)


@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.
    
    Attributes:
        rotation_range: Random rotation in degrees.
        width_shift_range: Horizontal shift as fraction.
        height_shift_range: Vertical shift as fraction.
        shear_range: Shear intensity.
        zoom_range: Random zoom range.
        horizontal_flip: Enable horizontal flipping.
        fill_mode: Fill mode for empty pixels.
    """
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    fill_mode: str = "nearest"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ImageDataGenerator."""
        return {
            "rotation_range": self.rotation_range,
            "width_shift_range": self.width_shift_range,
            "height_shift_range": self.height_shift_range,
            "shear_range": self.shear_range,
            "zoom_range": self.zoom_range,
            "horizontal_flip": self.horizontal_flip,
            "fill_mode": self.fill_mode,
            "rescale": 1.0 / 255.0
        }


class DataPipeline(BaseService):
    """
    Data pipeline for CNN training.
    
    Handles:
    - Loading images from directories
    - Data augmentation for training
    - Train/validation splitting
    - Batching and prefetching
    
    The pipeline expects directory structure:
    ```
    data_dir/
        class_1/
            image1.jpg
            image2.jpg
        class_2/
            image3.jpg
    ```
    
    Example:
        >>> pipeline = DataPipeline()
        >>> train_gen, val_gen = pipeline.create_generators(
        ...     data_dir="./data/images",
        ...     validation_split=0.2
        ... )
    """
    
    def __init__(
        self,
        image_config: Optional[ImageConfig] = None,
        augmentation_config: Optional[AugmentationConfig] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            image_config: Image processing configuration.
            augmentation_config: Data augmentation configuration.
            
        Raises:
            ConfigurationError: If TensorFlow is not available.
        """
        super().__init__("DataPipeline")
        
        if not TF_AVAILABLE:
            raise ConfigurationError(
                "tensorflow",
                "TensorFlow is required for data pipeline"
            )
        
        config = get_config()
        self.image_config = image_config or config.image
        self.augmentation_config = augmentation_config or AugmentationConfig()
        
        self._train_generator = None
        self._validation_generator = None
        self._class_indices: Dict[str, int] = {}
        
        self._mark_initialized()
    
    def create_generators(
        self,
        data_dir: str,
        validation_split: float = 0.2,
        batch_size: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """
        Create training and validation data generators.
        
        Args:
            data_dir: Path to data directory.
            validation_split: Fraction for validation.
            batch_size: Batch size (uses config default if not specified).
            
        Returns:
            Tuple of (train_generator, validation_generator).
            
        Raises:
            ValidationError: If data directory doesn't exist.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValidationError("data_dir", f"Directory not found: {data_dir}")
        
        batch_size = batch_size or self.image_config.batch_size
        
        # Training generator with augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            **self.augmentation_config.to_dict(),
            validation_split=validation_split
        )
        
        # Validation generator without augmentation
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=validation_split
        )
        
        self._train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_config.target_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True
        )
        
        self._validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_config.target_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )
        
        self._class_indices = self._train_generator.class_indices
        
        self.logger.info(
            f"Created generators: "
            f"{self._train_generator.samples} train, "
            f"{self._validation_generator.samples} val, "
            f"{len(self._class_indices)} classes"
        )
        
        return self._train_generator, self._validation_generator
    
    def create_tf_dataset(
        self,
        data_dir: str,
        validation_split: float = 0.2,
        batch_size: Optional[int] = None,
        seed: int = 42
    ) -> Tuple[Any, Any]:
        """
        Create tf.data.Dataset for efficient data loading.
        
        This is preferred over generators for large datasets
        as it enables better prefetching and parallelization.
        
        Args:
            data_dir: Path to data directory.
            validation_split: Fraction for validation.
            batch_size: Batch size.
            seed: Random seed for splitting.
            
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        batch_size = batch_size or self.image_config.batch_size
        
        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=self.image_config.target_size,
            batch_size=batch_size
        )
        
        val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=self.image_config.target_size,
            batch_size=batch_size
        )
        
        self._class_indices = {
            name: idx for idx, name in enumerate(train_ds.class_names)
        }
        
        # Normalize pixel values
        normalization = keras.layers.Rescaling(1.0 / 255)
        train_ds = train_ds.map(lambda x, y: (normalization(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization(x), y))
        
        # Optimize for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
        
        self.logger.info(f"Created tf.data.Dataset with {len(self._class_indices)} classes")
        
        return train_ds, val_ds
    
    def get_class_indices(self) -> Dict[str, int]:
        """Get mapping of class names to indices."""
        return self._class_indices.copy()
    
    def get_index_to_class(self) -> Dict[int, str]:
        """Get mapping of indices to class names."""
        return {v: k for k, v in self._class_indices.items()}
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self._class_indices)
    
    @property
    def train_samples(self) -> int:
        """Number of training samples."""
        if self._train_generator:
            return self._train_generator.samples
        return 0
    
    @property
    def validation_samples(self) -> int:
        """Number of validation samples."""
        if self._validation_generator:
            return self._validation_generator.samples
        return 0
    
    def health_check(self) -> dict:
        """Check pipeline health."""
        base = super().health_check()
        base["tensorflow_available"] = TF_AVAILABLE
        base["num_classes"] = self.get_num_classes()
        base["train_samples"] = self.train_samples
        base["validation_samples"] = self.validation_samples
        return base
