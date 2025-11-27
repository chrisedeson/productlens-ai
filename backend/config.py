"""
Configuration management for ProductLens AI backend.
Loads environment variables and provides centralized config access.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Pinecone Settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "product-recommendations")
    PINECONE_HOST: str = os.getenv("PINECONE_HOST", "")
    
    # Flask Settings
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    
    # Data Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    CLEANED_DATA_DIR: str = os.path.join(DATA_DIR, "cleaned")
    IMAGES_DIR: str = os.path.join(DATA_DIR, "images")
    TRAINED_MODELS_DIR: str = os.path.join(BASE_DIR, "trained_models")
    
    # Dataset Files
    RAW_DATASET_PATH: str = os.path.join(RAW_DATA_DIR, "dataset.csv")
    CLEANED_DATASET_PATH: str = os.path.join(CLEANED_DATA_DIR, "cleaned_dataset.csv")
    CNN_TRAIN_DATA_PATH: str = os.path.join(RAW_DATA_DIR, "CNN_Model_Train_Data.csv")
    
    # CNN Model Settings
    CNN_MODEL_PATH: str = os.path.join(TRAINED_MODELS_DIR, "cnn_product_classifier.h5")
    CNN_IMAGE_SIZE: tuple = (224, 224)
    CNN_BATCH_SIZE: int = 32
    CNN_EPOCHS: int = 50
    
    # Embedding Dimensions (text-embedding-3-small produces 1536 dimensions)
    EMBEDDING_DIMENSIONS: int = 1536
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        required = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("PINECONE_API_KEY", cls.PINECONE_API_KEY),
            ("PINECONE_HOST", cls.PINECONE_HOST),
        ]
        
        missing = [name for name, value in required if not value]
        
        if missing:
            print(f"Missing required configuration: {', '.join(missing)}")
            return False
        return True


# Create a singleton instance
config = Config()
