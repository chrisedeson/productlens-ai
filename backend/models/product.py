"""
Product data model for ProductLens AI.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Product:
    """Represents a product in the e-commerce dataset."""
    
    stock_code: str
    description: str
    unit_price: float
    country: str
    quantity: Optional[int] = None
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None
    customer_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert product to dictionary for API responses."""
        return {
            "stock_code": self.stock_code,
            "description": self.description,
            "unit_price": self.unit_price,
            "country": self.country,
        }
    
    def to_metadata(self) -> dict:
        """Convert product to metadata format for Pinecone storage."""
        return {
            "stock_code": str(self.stock_code),
            "description": str(self.description),
            "unit_price": float(self.unit_price) if self.unit_price else 0.0,
            "country": str(self.country),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Product":
        """Create a Product instance from a dictionary."""
        return cls(
            stock_code=data.get("stock_code", ""),
            description=data.get("description", ""),
            unit_price=float(data.get("unit_price", 0.0)),
            country=data.get("country", ""),
            quantity=data.get("quantity"),
            invoice_no=data.get("invoice_no"),
            invoice_date=data.get("invoice_date"),
            customer_id=data.get("customer_id"),
        )
    
    @classmethod
    def from_pinecone_match(cls, match: dict) -> "Product":
        """Create a Product instance from a Pinecone match result."""
        metadata = match.get("metadata", {})
        return cls(
            stock_code=metadata.get("stock_code", ""),
            description=metadata.get("description", ""),
            unit_price=float(metadata.get("unit_price", 0.0)),
            country=metadata.get("country", ""),
        )
