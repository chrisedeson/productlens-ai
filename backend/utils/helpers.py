"""
Helper functions for ProductLens AI.
"""

import re
from typing import Optional


def format_price(price: float, currency: str = "$") -> str:
    """
    Format a price value for display.
    
    Args:
        price: The numeric price value.
        currency: Currency symbol to use.
        
    Returns:
        Formatted price string.
    """
    if price is None or price < 0:
        return f"{currency}0.00"
    return f"{currency}{price:.2f}"


def sanitize_text(text: str) -> str:
    """
    Clean and sanitize text by removing noise characters.
    
    Args:
        text: Input text to sanitize.
        
    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove common noise characters found in the dataset
    noise_chars = ["ö", "^", "ä", "☺️", "XxY", "Ww", "&", "#", "@", "$"]
    
    result = text
    for char in noise_chars:
        result = result.replace(char, "")
    
    # Remove multiple spaces
    result = re.sub(r"\s+", " ", result)
    
    # Strip leading/trailing whitespace
    result = result.strip()
    
    return result


def clean_stock_code(code: str) -> str:
    """
    Clean a stock code by removing noise characters.
    
    Args:
        code: Raw stock code from dataset.
        
    Returns:
        Cleaned stock code.
    """
    if not code or not isinstance(code, str):
        return ""
    
    # Remove common noise prefixes/suffixes
    result = str(code).strip()
    result = re.sub(r"[öä^]", "", result)
    
    return result


def clean_price(price_str: str) -> Optional[float]:
    """
    Parse and clean a price string.
    
    Args:
        price_str: Price value that may contain noise.
        
    Returns:
        Float price value or None if invalid.
    """
    if price_str is None:
        return None
    
    try:
        # Convert to string and remove noise
        price_str = str(price_str)
        price_str = re.sub(r"[Ww$£€,]", "", price_str)
        price_str = price_str.strip()
        
        if not price_str:
            return None
            
        return float(price_str)
    except (ValueError, TypeError):
        return None


def clean_country(country: str) -> str:
    """
    Clean a country name by removing noise.
    
    Args:
        country: Raw country value from dataset.
        
    Returns:
        Cleaned country name.
    """
    if not country or not isinstance(country, str):
        return "Unknown"
    
    # Remove noise characters
    result = sanitize_text(country)
    
    # Remove common noise patterns
    result = re.sub(r"[☺️]", "", result)
    
    return result.strip() if result.strip() else "Unknown"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate.
        max_length: Maximum allowed length.
        
    Returns:
        Truncated text with ellipsis if needed.
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."
