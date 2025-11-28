"""
ProductLens AI - Main Flask Application

An intelligent e-commerce product recommendation system with:
- Natural language product search (Endpoint 1)
- OCR-based handwritten query processing (Endpoint 2)
- CNN-based product image detection (Endpoint 3)
"""

import os
import logging
from flask import Flask, render_template, jsonify
from flask_cors import CORS

from config import config
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from services.llm_service import LLMService
from services.recommendation_service import RecommendationService
from services.ocr_service import OCRService

from routes.recommendation_routes import recommendation_bp, init_recommendation_routes
from routes.ocr_routes import ocr_bp, init_ocr_routes
from routes.image_routes import image_bp, init_image_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Application factory for creating the Flask app.
    
    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)
    
    # Enable CORS for frontend
    # Allows localhost for development and Vercel domains for production
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://productlens-ai.vercel.app",
        "https://productlens-fn44jydoq-christopher-edesons-projects.vercel.app",
    ]
    
    # Add production frontend URL if set
    frontend_url = os.environ.get("FRONTEND_URL")
    if frontend_url:
        allowed_origins.append(frontend_url)
    
    # Allow all Vercel preview deployments
    CORS(app, origins=allowed_origins, resources={
        r"/*": {
            "origins": allowed_origins,
            "allow_headers": ["Content-Type", "Authorization"],
            "methods": ["GET", "POST", "OPTIONS"]
        }
    })
    
    # Validate configuration
    if not config.validate():
        logger.warning("Some configuration values are missing. Some features may not work.")
    
    # Initialize services
    try:
        # Core services
        embedding_service = EmbeddingService(
            api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL
        )
        
        vector_service = VectorService(
            api_key=config.PINECONE_API_KEY,
            index_name=config.PINECONE_INDEX_NAME,
            host=config.PINECONE_HOST
        )
        
        llm_service = LLMService(
            api_key=config.OPENAI_API_KEY
        )
        
        recommendation_service = RecommendationService(
            embedding_service=embedding_service,
            vector_service=vector_service,
            llm_service=llm_service
        )
        
        # OCR service
        try:
            ocr_service = OCRService(openai_api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"OCR service not available: {e}")
            ocr_service = None
        
        # Initialize route handlers with services
        init_recommendation_routes(recommendation_service)
        
        if ocr_service:
            init_ocr_routes(ocr_service, recommendation_service)
        
        # Image routes (CNN service is lazy-loaded in the route)
        init_image_routes(recommendation_service, openai_api_key=config.OPENAI_API_KEY)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    
    # Register blueprints
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(ocr_bp)
    app.register_blueprint(image_bp)
    
    # Health check endpoint
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        # Check CNN service status lazily
        cnn_available = False
        try:
            from services.image_classification_service import ImageClassificationService
            cnn_svc = ImageClassificationService()
            cnn_available = cnn_svc.model_loaded
        except Exception:
            pass
        
        return jsonify({
            "status": "healthy",
            "services": {
                "recommendation": True,
                "ocr": ocr_service is not None,
                "cnn": cnn_available
            }
        })
    
    # Sample response endpoint (from original app.py)
    @app.route("/sample_response", methods=["GET"])
    def sample_response():
        """Sample response page showing expected output format."""
        return render_template("sample_response.html")
    
    # Root endpoint
    @app.route("/", methods=["GET"])
    def index():
        """API information endpoint."""
        return jsonify({
            "name": "ProductLens AI",
            "version": "1.0.0",
            "description": "E-commerce product recommendation system",
            "endpoints": {
                "/product-recommendation": "POST - Natural language product search",
                "/ocr-query": "POST - Handwritten query processing",
                "/image-product-search": "POST - Product image detection",
                "/health": "GET - Health check",
                "/sample_response": "GET - Sample response format"
            }
        })
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
