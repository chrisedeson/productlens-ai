"""
Image Product Detection Routes for ProductLens AI.
Endpoint 3: Identify products from images using CNN.

Uses the ML inference module with EfficientNetV2 for product classification.
"""

from flask import Blueprint, request, jsonify
import logging
import os
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_bp = Blueprint("image", __name__)

# Services will be initialized lazily
_image_classifier = None
_recommendation_service = None


def get_image_classifier():
    """
    Lazy load the image classifier.
    
    Returns None if CNN is disabled or model fails to load.
    """
    global _image_classifier
    
    # Check if CNN is disabled
    if os.environ.get("DISABLE_CNN", "").lower() in ("true", "1", "yes"):
        logger.info("CNN is disabled via DISABLE_CNN environment variable")
        return None
    
    if _image_classifier is None:
        try:
            from ml.inference.image_classifier import ImageClassifier
            _image_classifier = ImageClassifier()
            logger.info("Image classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image classifier: {e}")
            return None
    
    return _image_classifier


def init_image_routes(rec_svc):
    """Initialize routes with recommendation service."""
    global _recommendation_service
    _recommendation_service = rec_svc


@image_bp.route("/image-product-search", methods=["POST"])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    
    Uses the CNN model trained from scratch (no pre-trained models) to classify
    product images into one of 10 categories, then returns similar products.
    
    Input: Form data containing 'product_image' (image file upload)
    Output: JSON with 'products', 'response', 'predicted_class', and 'predictions'
    """
    try:
        # Check if image was uploaded
        if "product_image" not in request.files:
            return jsonify({
                "products": [],
                "response": "Please upload a product image.",
                "predicted_class": "",
                "predictions": []
            }), 400
        
        product_image = request.files["product_image"]
        
        if product_image.filename == "":
            return jsonify({
                "products": [],
                "response": "No image file selected.",
                "predicted_class": "",
                "predictions": []
            }), 400
        
        logger.info(f"Received image product search: {product_image.filename}")
        
        # Read image data
        image_data = product_image.read()
        
        # Get classifier (lazy loaded)
        classifier = get_image_classifier()
        
        if classifier is None:
            return jsonify({
                "products": [],
                "response": "Image classification is temporarily unavailable. Please try text search instead.",
                "predicted_class": "",
                "predictions": []
            }), 503
        
        # Use CNN to classify the image
        predictions = classifier.classify_from_bytes(image_data, top_k=5)
        
        if not predictions:
            return jsonify({
                "products": [],
                "response": "Could not identify the product in the image. Please try with a clearer image.",
                "predicted_class": "",
                "predictions": []
            })
        
        # Get top prediction
        top_prediction = predictions[0]
        predicted_class = top_prediction['description']
        confidence = top_prediction['confidence']
        
        logger.info(f"Predicted class: {predicted_class} (confidence: {confidence:.2%})")
        
        # Get similar products from vector database using the predicted class
        products = []
        response = f"Based on the image, I identified this as: **{predicted_class}** (confidence: {confidence:.1%})."
        
        if _recommendation_service:
            try:
                # Search for similar products using the class description
                search_results, rec_response = _recommendation_service.recommend(predicted_class, top_k=5)
                if search_results:
                    products = search_results
                    response += f" {rec_response}"
                else:
                    response += " No matching products found in the catalog."
            except Exception as e:
                logger.warning(f"Could not get recommendations: {e}")
                response += " Could not retrieve product recommendations."
        else:
            response += " Recommendation service not available."
        
        return jsonify({
            "products": products,
            "response": response,
            "predicted_class": predicted_class,
            "predictions": predictions
        })
        
    except Exception as e:
        logger.error(f"Error in image product search: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "products": [],
            "response": f"An error occurred while processing your image: {str(e)}",
            "predicted_class": "",
            "predictions": []
        }), 500


@image_bp.route("/model-info", methods=["GET"])
def get_model_info():
    """Get information about the CNN model."""
    try:
        classifier = get_image_classifier()
        if classifier is None:
            return jsonify({
                "status": "unavailable",
                "message": "Image classification is disabled or model not loaded"
            })
        return jsonify(classifier.get_model_info())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
