"""
Image Product Detection Routes for ProductLens AI.
Endpoint 3: Identify products from images using CNN.
"""

from flask import Blueprint, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_bp = Blueprint("image", __name__)

# Services will be injected from app.py
cnn_service = None
recommendation_service = None


def init_image_routes(cnn_svc, rec_svc):
    """Initialize routes with services."""
    global cnn_service, recommendation_service
    cnn_service = cnn_svc
    recommendation_service = rec_svc


@image_bp.route("/image-product-search", methods=["POST"])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    
    Input: Form data containing 'product_image' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products', 'response', and 'predicted_class'.
    """
    try:
        # Check if image was uploaded
        if "product_image" not in request.files:
            return jsonify({
                "products": [],
                "response": "Please upload a product image.",
                "predicted_class": ""
            }), 400
        
        product_image = request.files["product_image"]
        
        if product_image.filename == "":
            return jsonify({
                "products": [],
                "response": "No image file selected.",
                "predicted_class": ""
            }), 400
        
        logger.info(f"Received image product search: {product_image.filename}")
        
        # Read image data
        image_data = product_image.read()
        
        # Use CNN to predict product class
        predicted_class, confidence = cnn_service.predict(image_data)
        
        if not predicted_class:
            return jsonify({
                "products": [],
                "response": "Could not identify the product in the image. Please try with a clearer image.",
                "predicted_class": ""
            })
        
        logger.info(f"Predicted class: {predicted_class} (confidence: {confidence:.2f})")
        
        # Get similar products from vector database
        products, response = recommendation_service.recommend_from_class(
            predicted_class, 
            top_k=5
        )
        
        return jsonify({
            "products": products,
            "response": response,
            "predicted_class": predicted_class
        })
        
    except Exception as e:
        logger.error(f"Error in image product search: {e}")
        return jsonify({
            "products": [],
            "response": "An error occurred while processing your image.",
            "predicted_class": ""
        }), 500
