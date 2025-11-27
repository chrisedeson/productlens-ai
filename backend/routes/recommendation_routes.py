"""
Product Recommendation Routes for ProductLens AI.
Endpoint 1: Natural language product recommendations.
"""

from flask import Blueprint, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

recommendation_bp = Blueprint("recommendation", __name__)

# Service instances will be injected from app.py
recommendation_service = None


def init_recommendation_routes(service):
    """Initialize routes with the recommendation service."""
    global recommendation_service
    recommendation_service = service


@recommendation_bp.route("/product-recommendation", methods=["POST"])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    
    Input: Form data containing 'query' (string).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    try:
        # Get query from form data or JSON
        if request.is_json:
            data = request.get_json()
            query = data.get("query", "")
        else:
            query = request.form.get("query", "")
        
        if not query:
            return jsonify({
                "products": [],
                "response": "Please provide a search query."
            }), 400
        
        logger.info(f"Received product recommendation query: {query[:50]}...")
        
        # Get recommendations
        products, response = recommendation_service.recommend(query, top_k=5)
        
        return jsonify({
            "products": products,
            "response": response
        })
        
    except Exception as e:
        logger.error(f"Error in product recommendation: {e}")
        return jsonify({
            "products": [],
            "response": "An error occurred while processing your request."
        }), 500
