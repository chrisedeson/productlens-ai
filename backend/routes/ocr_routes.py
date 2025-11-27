"""
OCR Query Routes for ProductLens AI.
Endpoint 2: Process handwritten queries from images.
"""

from flask import Blueprint, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ocr_bp = Blueprint("ocr", __name__)

# Services will be injected from app.py
ocr_service = None
recommendation_service = None


def init_ocr_routes(ocr_svc, rec_svc):
    """Initialize routes with services."""
    global ocr_service, recommendation_service
    ocr_service = ocr_svc
    recommendation_service = rec_svc


@ocr_bp.route("/ocr-query", methods=["POST"])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    
    Input: Form data containing 'image_data' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products', 'response', and 'extracted_text'.
    """
    try:
        # Check if image was uploaded
        if "image_data" not in request.files:
            return jsonify({
                "products": [],
                "response": "Please upload an image file.",
                "extracted_text": ""
            }), 400
        
        image_file = request.files["image_data"]
        
        if image_file.filename == "":
            return jsonify({
                "products": [],
                "response": "No image file selected.",
                "extracted_text": ""
            }), 400
        
        logger.info(f"Received OCR query with image: {image_file.filename}")
        
        # Read image data
        image_data = image_file.read()
        
        # Extract text using OCR
        extracted_text = ocr_service.extract_text(image_data)
        
        if not extracted_text or not extracted_text.strip():
            return jsonify({
                "products": [],
                "response": "Could not extract any text from the image. Please try with a clearer image.",
                "extracted_text": ""
            })
        
        logger.info(f"Extracted text: {extracted_text[:50]}...")
        
        # Get recommendations based on extracted text
        products, response = recommendation_service.recommend_from_text(
            extracted_text, 
            top_k=5,
            is_ocr=True
        )
        
        return jsonify({
            "products": products,
            "response": response,
            "extracted_text": extracted_text
        })
        
    except Exception as e:
        logger.error(f"Error in OCR query: {e}")
        return jsonify({
            "products": [],
            "response": "An error occurred while processing your image.",
            "extracted_text": ""
        }), 500
