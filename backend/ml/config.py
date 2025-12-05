"""
ML Configuration Module for ProductLens AI.

Centralized configuration for machine learning pipelines.
Contains hyperparameters, paths, and model configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "ml_artifacts"
MODELS_DIR = PROJECT_ROOT / "models"  # Trained model files


@dataclass(frozen=True)
class ImageConfig:
    """
    Configuration for image processing.
    
    Attributes:
        target_size: Target image dimensions (height, width).
        channels: Number of color channels (3 for RGB).
        batch_size: Batch size for training and inference.
        rescale: Pixel value rescaling factor.
        valid_extensions: Allowed image file extensions.
    """
    target_size: tuple = (224, 224)
    channels: int = 3
    batch_size: int = 32
    rescale: float = 1.0 / 255.0
    valid_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")
    
    @property
    def input_shape(self) -> tuple:
        """Get full input shape for model."""
        return (*self.target_size, self.channels)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for CNN model training.
    
    Attributes:
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        validation_split: Fraction of data for validation.
        early_stopping_patience: Epochs to wait before stopping.
        reduce_lr_patience: Epochs to wait before reducing LR.
        reduce_lr_factor: Factor to reduce learning rate by.
        min_lr: Minimum learning rate.
    """
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    min_lr: float = 1e-7
    
    # Data augmentation settings
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    fill_mode: str = "nearest"


@dataclass(frozen=True)
class CNNModelConfig:
    """
    Configuration for the CNN architecture.
    
    Attributes:
        base_model: Name of the pretrained base model.
        freeze_base: Whether to freeze base model weights.
        dense_units: Units in dense layers.
        dropout_rate: Dropout rate for regularization.
        l2_regularization: L2 regularization factor.
    """
    base_model: str = "MobileNetV2"
    freeze_base: bool = True
    dense_units: List[int] = field(default_factory=lambda: [512, 256])
    dropout_rate: float = 0.5
    l2_regularization: float = 0.01
    
    # Available base models
    AVAILABLE_BASE_MODELS = [
        "MobileNetV2",
        "ResNet50",
        "VGG16",
        "InceptionV3",
        "EfficientNetB0"
    ]


@dataclass
class DataPaths:
    """
    Configuration for data and model paths.
    
    Attributes:
        raw_data: Path to raw dataset CSV.
        images_dir: Directory for downloaded images.
        processed_dir: Directory for processed data.
        model_dir: Directory for saved models.
        logs_dir: Directory for training logs.
    """
    raw_data: Path = DATA_DIR / "products.csv"
    images_dir: Path = DATA_DIR / "images"
    processed_dir: Path = DATA_DIR / "processed"
    train_dir: Path = field(default=None)
    validation_dir: Path = field(default=None)
    model_dir: Path = MODELS_DIR
    logs_dir: Path = ARTIFACTS_DIR / "logs"
    checkpoints_dir: Path = ARTIFACTS_DIR / "checkpoints"
    
    def __post_init__(self):
        """Set derived paths after initialization."""
        if self.train_dir is None:
            self.train_dir = self.images_dir / "train"
        if self.validation_dir is None:
            self.validation_dir = self.images_dir / "validation"
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [
            self.images_dir,
            self.processed_dir,
            self.train_dir,
            self.validation_dir,
            self.model_dir,
            self.logs_dir,
            self.checkpoints_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class HuggingFaceConfig:
    """
    Configuration for Hugging Face model hosting.
    
    Attributes:
        repo_id: Hugging Face repository ID.
        model_filename: Name of the model file in the repo.
        revision: Model version/revision.
    """
    repo_id: str = "chrisedeson/productlens-cnn-model"
    model_filename: str = "product_classifier.keras"
    revision: str = "main"
    
    @property
    def token(self) -> Optional[str]:
        """Get HF token from environment."""
        return os.environ.get("HF_TOKEN")


@dataclass(frozen=True)
class InferenceConfig:
    """
    Configuration for model inference.
    
    Attributes:
        confidence_threshold: Minimum confidence for predictions.
        top_k: Number of top predictions to return.
        batch_size: Batch size for batch inference.
    """
    confidence_threshold: float = 0.5
    top_k: int = 5
    batch_size: int = 16


@dataclass
class ProductCategories:
    """
    Product category mappings for the classifier.
    
    Maps category indices to human-readable names.
    """
    _categories: Dict[int, str] = field(default_factory=dict)
    
    def load_from_file(self, path: Path) -> None:
        """Load category mappings from a JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
            self._categories = {int(k): v for k, v in data.items()}
    
    def get_category(self, index: int) -> str:
        """Get category name by index."""
        return self._categories.get(index, f"Category_{index}")
    
    def get_index(self, category: str) -> Optional[int]:
        """Get category index by name."""
        for idx, name in self._categories.items():
            if name == category:
                return idx
        return None
    
    @property
    def num_classes(self) -> int:
        """Get total number of categories."""
        return len(self._categories)
    
    @property
    def all_categories(self) -> List[str]:
        """Get list of all category names."""
        return list(self._categories.values())


@dataclass
class MLConfig:
    """
    Master configuration combining all ML settings.
    
    This is the main entry point for accessing ML configuration.
    
    Example:
        >>> config = MLConfig()
        >>> print(config.training.epochs)
        >>> print(config.paths.model_dir)
    """
    image: ImageConfig = field(default_factory=ImageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: CNNModelConfig = field(default_factory=CNNModelConfig)
    paths: DataPaths = field(default_factory=DataPaths)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    categories: ProductCategories = field(default_factory=ProductCategories)
    
    def __post_init__(self):
        """Initialize paths and categories."""
        self.paths.ensure_directories()
        
        # Try to load categories if file exists
        categories_file = self.paths.model_dir / "categories.json"
        if categories_file.exists():
            self.categories.load_from_file(categories_file)


# Default configuration instance
default_config = MLConfig()


def get_config() -> MLConfig:
    """
    Get the default ML configuration.
    
    Returns:
        MLConfig instance with default settings.
    """
    return default_config


def create_config(**overrides) -> MLConfig:
    """
    Create a custom ML configuration with overrides.
    
    Args:
        **overrides: Configuration overrides.
        
    Returns:
        MLConfig instance with custom settings.
        
    Example:
        >>> config = create_config(
        ...     training=TrainingConfig(epochs=100, learning_rate=0.0001)
        ... )
    """
    return MLConfig(**overrides)
