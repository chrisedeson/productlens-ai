"""
LLM Service for ProductLens AI.

Handles OpenAI GPT interactions for generating product recommendations.
"""

from typing import List, Dict, Any, Optional
import logging
import json

from openai import OpenAI, OpenAIError, RateLimitError as OpenAIRateLimitError

from core.base_service import BaseService, RetryMixin
from core.exceptions import (
    ExternalAPIError,
    ValidationError,
    ConfigurationError,
    RateLimitError
)

logging.basicConfig(level=logging.INFO)


class LLMService(BaseService, RetryMixin):
    """
    Service for generating AI-powered product recommendations.
    
    Uses OpenAI GPT-4 to analyze user queries and format search results
    into natural language recommendations with explanations.
    
    The service:
    - Processes user search queries
    - Formats product matches into recommendations
    - Generates personalized explanations
    - Handles context about purchase occasions
    
    Attributes:
        model: OpenAI model identifier (default: gpt-4).
        max_tokens: Maximum response tokens.
    
    Example:
        >>> service = LLMService(api_key="...")
        >>> response = service.generate_recommendations(
        ...     query="gift for mom",
        ...     products=[{"description": "Scented candle set", ...}]
        ... )
    """
    
    DEFAULT_MODEL = "gpt-4"
    DEFAULT_MAX_TOKENS = 150
    DEFAULT_TEMPERATURE = 0.7
    
    SYSTEM_PROMPT = """You are a helpful shopping assistant for an e-commerce platform.
Your role is to recommend products based on user queries and explain why each product 
is a good match. Consider the context, occasion, and user preferences when making 
recommendations.

Guidelines:
- Be friendly and conversational
- Explain why each product matches the user's needs
- Highlight key features and benefits
- Consider gift-giving context if mentioned
- Keep recommendations concise but informative"""
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        """
        Initialize the LLM Service.
        
        Args:
            api_key: OpenAI API key.
            model: Model to use for generation.
            max_tokens: Maximum tokens in response.
            
        Raises:
            ValidationError: If API key is missing.
            ConfigurationError: If OpenAI client initialization fails.
        """
        super().__init__("LLMService")
        
        if not api_key:
            raise ValidationError("api_key", "OpenAI API key is required")
        
        self.model = model
        self.max_tokens = max_tokens
        
        try:
            self._client = OpenAI(api_key=api_key)
            self._mark_initialized()
            self.logger.info(f"LLM Service initialized with model: {model}")
        except OpenAIError as e:
            raise ConfigurationError(
                "openai_client",
                f"Failed to initialize OpenAI client: {str(e)}"
            )
    
    def generate_recommendations(
        self,
        query: str,
        products: List[Dict[str, Any]],
        max_recommendations: int = 5,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate product recommendations with explanations.
        
        Args:
            query: User's search query.
            products: List of product matches from vector search.
            max_recommendations: Maximum number of products to recommend.
            context: Optional additional context.
            
        Returns:
            Dictionary with recommendations and AI-generated explanations.
            
        Example:
            >>> result = service.generate_recommendations(
            ...     query="birthday gift for dad",
            ...     products=[...],
            ...     context="budget: medium"
            ... )
            >>> print(result["explanation"])
        """
        if not query:
            raise ValidationError("query", "Query cannot be empty")
        
        if not products:
            return {
                "success": True,
                "query": query,
                "recommendations": [],
                "explanation": "I couldn't find any products matching your search. "
                              "Try using different keywords or browse our categories."
            }
        
        # Limit products to process
        products_to_process = products[:max_recommendations]
        
        # Build the prompt
        prompt = self._build_recommendation_prompt(
            query=query,
            products=products_to_process,
            context=context
        )
        
        try:
            response = self._with_retry(
                lambda: self._call_openai(prompt),
                max_retries=3,
                exceptions=(OpenAIError,)
            )
            
            return {
                "success": True,
                "query": query,
                "recommendations": self._format_recommendations(products_to_process),
                "explanation": response,
                "model": self.model
            }
            
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                api_name="OpenAI",
                retry_after=60
            )
        except OpenAIError as e:
            raise ExternalAPIError(
                api_name="OpenAI",
                message=f"Failed to generate recommendations: {str(e)}",
                original_error=e
            )
    
    def _call_openai(self, prompt: str) -> str:
        """
        Make a call to OpenAI API.
        
        Args:
            prompt: User prompt for the model.
            
        Returns:
            Generated response text.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.DEFAULT_TEMPERATURE
        )
        
        return response.choices[0].message.content
    
    def _build_recommendation_prompt(
        self,
        query: str,
        products: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> str:
        """
        Build the recommendation prompt.
        
        Args:
            query: User's search query.
            products: Product matches.
            context: Additional context.
            
        Returns:
            Formatted prompt string.
        """
        product_list = "\n".join([
            f"- {p.get('description', 'Unknown product')} "
            f"(Stock: {p.get('stock_code', 'N/A')}, "
            f"Price: £{p.get('unit_price', 'N/A')})"
            for p in products
        ])
        
        prompt = f"""User searched for: "{query}"

Found {len(products)} matching products.

Give a brief 1-2 sentence friendly response about these results. Do NOT list the products - they're shown in a table."""
        
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        return prompt
    
    def _format_recommendations(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format products for API response.
        
        Args:
            products: Raw product data.
            
        Returns:
            Formatted product list.
        """
        return [
            {
                "stock_code": p.get("stock_code", ""),
                "description": p.get("description", ""),
                "unit_price": p.get("unit_price"),
                "country": p.get("country", ""),
                "similarity_score": p.get("score", 0)
            }
            for p in products
        ]
    
    def generate_product_summary(
        self,
        product: Dict[str, Any]
    ) -> str:
        """
        Generate a brief AI summary for a single product.
        
        Args:
            product: Product data dictionary.
            
        Returns:
            AI-generated product summary.
        """
        description = product.get("description", "Product")
        
        prompt = f"""Generate a brief, appealing 1-2 sentence summary for this product:
{description}

Make it engaging and highlight potential uses or gift occasions."""
        
        try:
            return self._call_openai(prompt)
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return description
    
    def analyze_search_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the intent behind a user's search query.
        
        Args:
            query: User's search query.
            
        Returns:
            Analysis of search intent (category, occasion, etc.).
        """
        prompt = f"""Analyze this e-commerce search query and identify:
1. Product category (if any)
2. Occasion (gift, personal use, etc.)
3. Key attributes mentioned
4. Suggested search filters

Query: "{query}"

Respond in JSON format with keys: category, occasion, attributes, filters"""
        
        try:
            response = self._call_openai(prompt)
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_analysis": response}
        except Exception as e:
            self.logger.error(f"Error analyzing intent: {e}")
            return {"error": str(e)}
    
    # ----- OCR and Image Response Methods -----
    
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
        if not extracted_text or not extracted_text.strip():
            return "I couldn't extract any text from the image. Please try with a clearer image."
        
        prompt = f"""You are a helpful e-commerce assistant. 
A customer uploaded a handwritten shopping list or note.

Extracted text from their handwriting: "{extracted_text}"

We found {len(products)} matching products in our catalog.

Write a brief, friendly response (2-3 sentences) that:
1. Confirms what you understood from their handwriting
2. Mentions that you found matching products
3. Is conversational and helpful"""
        
        try:
            return self._call_openai(prompt)
        except Exception as e:
            self.logger.error(f"Error generating OCR response: {e}")
            return f"I read your note and found products matching: {extracted_text[:100]}"
    
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
        
        prompt = f"""You are a helpful e-commerce assistant. 
A customer uploaded an image of a product.

Our system identified the product as: "{predicted_class}"

We found {len(products)} similar products in our catalog.

Write a brief, friendly response (2-3 sentences) that:
1. Confirms what product was detected in the image
2. Mentions that you found similar products
3. Is conversational and helpful"""
        
        try:
            return self._call_openai(prompt)
        except Exception as e:
            self.logger.error(f"Error generating image detection response: {e}")
            return f"I identified this as a {predicted_class}. Here are similar products from our catalog."
    
    def health_check(self) -> dict:
        """
        Check if the LLM service is operational.
        
        Returns:
            Health status dictionary.
        """
        base_health = super().health_check()
        base_health["model"] = self.model
        
        try:
            # Simple test call
            test_response = self._client.models.list()
            base_health["openai_connected"] = True
            base_health["available_models"] = len(list(test_response))
        except Exception as e:
            base_health["status"] = "unhealthy"
            base_health["error"] = str(e)
        
        return base_health
