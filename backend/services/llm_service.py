"""
LLM Service for ProductLens AI.
Handles natural language response generation using OpenAI.
"""

from typing import List, Dict, Any
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """
    Service class for generating natural language responses using OpenAI.
    
    Generates human-friendly product descriptions and recommendations.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLMService.
        
        Args:
            api_key: OpenAI API key.
            model: Chat model to use.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_product_response(
        self, 
        query: str, 
        products: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a natural language response describing product recommendations.
        
        Args:
            query: The user's original query.
            products: List of product dictionaries with details.
            
        Returns:
            Natural language response string.
        """
        if not products:
            return "I couldn't find any products matching your query. Please try a different search term."
        
        # Build product list for the prompt
        product_descriptions = []
        for i, product in enumerate(products, 1):
            desc = product.get("description", "Unknown product")
            price = product.get("unit_price", 0)
            product_descriptions.append(f"{i}. {desc} (${price:.2f})")
        
        products_text = "\n".join(product_descriptions)
        
        prompt = f"""You are a helpful e-commerce assistant. A customer searched for: "{query}"

Based on their search, here are the matching products:
{products_text}

Write a brief, friendly response (2-3 sentences) that:
1. Acknowledges what they're looking for
2. Highlights the most relevant products
3. Is conversational and helpful

Do not list the products again - just describe them naturally in your response."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful e-commerce assistant that provides concise, friendly product recommendations."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Fallback response
            return self._generate_fallback_response(products)
    
    def _generate_fallback_response(self, products: List[Dict[str, Any]]) -> str:
        """Generate a simple fallback response without LLM."""
        if not products:
            return "No products found matching your search."
        
        product_names = [p.get("description", "product") for p in products[:3]]
        
        if len(product_names) == 1:
            return f"I found a great match for you: {product_names[0]}."
        else:
            names = ", ".join(product_names[:-1]) + f" and {product_names[-1]}"
            return f"Here are some products you might like: {names}."
    
    def generate_ocr_response(
        self, 
        extracted_text: str, 
        products: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a response for OCR-based queries.
        
        Args:
            extracted_text: Text extracted from the image via OCR.
            products: List of matching products.
            
        Returns:
            Natural language response.
        """
        if not extracted_text.strip():
            return "I couldn't extract any text from the image. Please try with a clearer image."
        
        return self.generate_product_response(extracted_text, products)
    
    def generate_image_detection_response(
        self, 
        predicted_class: str, 
        products: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a response for image-based product detection.
        
        Args:
            predicted_class: The predicted product class from CNN.
            products: List of similar products.
            
        Returns:
            Natural language response.
        """
        if not predicted_class:
            return "I couldn't identify the product in the image. Please try with a clearer image."
        
        prompt = f"""You are a helpful e-commerce assistant. A customer uploaded an image of a product.

Our system identified the product as: "{predicted_class}"

We found {len(products)} similar products in our catalog.

Write a brief, friendly response (2-3 sentences) that:
1. Confirms what product was detected in the image
2. Mentions that you found similar products
3. Is conversational and helpful"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful e-commerce assistant."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating image detection response: {e}")
            return f"I identified this as a {predicted_class}. Here are similar products from our catalog."
