"""
Data Cleaning Service for ProductLens AI.
Handles cleaning and preprocessing of the e-commerce dataset.
"""

import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Service class for cleaning and preprocessing the e-commerce dataset.
    
    Handles:
    - Removing special character noise
    - Handling missing values
    - Removing duplicates
    - Standardizing formats
    """
    
    # Noise characters found in the dataset
    NOISE_PATTERNS = {
        "ö": "",
        "^": "",
        "ä": "",
        "☺️": "",
        "XxY": "",
        "Ww": "",
        "&": "",
        "#": "",
        "@": "",
        "$": "",
    }
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the DataCleaner.
        
        Args:
            input_path: Path to the raw dataset CSV.
            output_path: Path where cleaned dataset will be saved.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.df: Optional[pd.DataFrame] = None
        self.stats = {
            "original_rows": 0,
            "cleaned_rows": 0,
            "duplicates_removed": 0,
            "nulls_removed": 0,
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load the raw dataset from CSV."""
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_csv(self.input_path, encoding="utf-8", low_memory=False)
        self.stats["original_rows"] = len(self.df)
        logger.info(f"Loaded {self.stats['original_rows']} rows")
        return self.df
    
    def _clean_text_column(self, text: str) -> str:
        """Clean a text value by removing noise characters."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        result = str(text)
        for noise, replacement in self.NOISE_PATTERNS.items():
            result = result.replace(noise, replacement)
        
        # Remove multiple spaces
        result = re.sub(r"\s+", " ", result)
        return result.strip()
    
    def _clean_numeric_column(self, value: str) -> Optional[float]:
        """Clean a numeric value by removing noise and parsing."""
        if pd.isna(value):
            return None
        
        try:
            # Convert to string and clean
            val_str = str(value)
            for noise in self.NOISE_PATTERNS.keys():
                val_str = val_str.replace(noise, "")
            val_str = val_str.strip()
            
            if not val_str:
                return None
            
            return float(val_str)
        except (ValueError, TypeError):
            return None
    
    def clean_invoice_no(self) -> "DataCleaner":
        """Clean the InvoiceNo column."""
        logger.info("Cleaning InvoiceNo column...")
        self.df["InvoiceNo"] = self.df["InvoiceNo"].apply(self._clean_text_column)
        return self
    
    def clean_stock_code(self) -> "DataCleaner":
        """Clean the StockCode column."""
        logger.info("Cleaning StockCode column...")
        self.df["StockCode"] = self.df["StockCode"].apply(self._clean_text_column)
        return self
    
    def clean_description(self) -> "DataCleaner":
        """Clean the Description column."""
        logger.info("Cleaning Description column...")
        self.df["Description"] = self.df["Description"].apply(self._clean_text_column)
        # Convert to uppercase for consistency
        self.df["Description"] = self.df["Description"].str.upper()
        return self
    
    def clean_quantity(self) -> "DataCleaner":
        """Clean the Quantity column."""
        logger.info("Cleaning Quantity column...")
        self.df["Quantity"] = self.df["Quantity"].apply(self._clean_numeric_column)
        # Convert to integer where possible
        self.df["Quantity"] = pd.to_numeric(self.df["Quantity"], errors="coerce")
        return self
    
    def clean_unit_price(self) -> "DataCleaner":
        """Clean the UnitPrice column."""
        logger.info("Cleaning UnitPrice column...")
        self.df["UnitPrice"] = self.df["UnitPrice"].apply(self._clean_numeric_column)
        return self
    
    def clean_customer_id(self) -> "DataCleaner":
        """Clean the CustomerID column."""
        logger.info("Cleaning CustomerID column...")
        self.df["CustomerID"] = self.df["CustomerID"].apply(self._clean_numeric_column)
        return self
    
    def clean_country(self) -> "DataCleaner":
        """Clean the Country column."""
        logger.info("Cleaning Country column...")
        self.df["Country"] = self.df["Country"].apply(self._clean_text_column)
        return self
    
    def clean_invoice_date(self) -> "DataCleaner":
        """Clean and parse the InvoiceDate column."""
        logger.info("Cleaning InvoiceDate column...")
        self.df["InvoiceDate"] = pd.to_datetime(
            self.df["InvoiceDate"], 
            errors="coerce",
            format="%Y-%m-%d %H:%M:%S"
        )
        return self
    
    def remove_duplicates(self) -> "DataCleaner":
        """Remove duplicate rows from the dataset."""
        logger.info("Removing duplicates...")
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        self.stats["duplicates_removed"] = before - len(self.df)
        logger.info(f"Removed {self.stats['duplicates_removed']} duplicates")
        return self
    
    def handle_missing_values(self) -> "DataCleaner":
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        before = len(self.df)
        
        # Drop rows where critical columns are missing
        critical_columns = ["StockCode", "Description"]
        self.df = self.df.dropna(subset=critical_columns)
        
        # Drop rows with empty descriptions after cleaning
        self.df = self.df[self.df["Description"].str.len() > 0]
        
        # Fill missing countries with "Unknown"
        self.df["Country"] = self.df["Country"].fillna("Unknown")
        
        # Fill missing prices with 0
        self.df["UnitPrice"] = self.df["UnitPrice"].fillna(0.0)
        
        self.stats["nulls_removed"] = before - len(self.df)
        logger.info(f"Removed {self.stats['nulls_removed']} rows with missing values")
        return self
    
    def filter_valid_data(self) -> "DataCleaner":
        """Filter out invalid data entries."""
        logger.info("Filtering invalid data...")
        before = len(self.df)
        
        # Remove rows with negative or zero prices (unless they're credits)
        self.df = self.df[self.df["UnitPrice"] >= 0]
        
        # Remove rows with very long or very short descriptions
        self.df = self.df[self.df["Description"].str.len() >= 3]
        self.df = self.df[self.df["Description"].str.len() <= 200]
        
        # Remove cancelled orders (InvoiceNo starting with 'C')
        self.df = self.df[~self.df["InvoiceNo"].str.startswith("C", na=False)]
        
        removed = before - len(self.df)
        logger.info(f"Filtered out {removed} invalid rows")
        return self
    
    def get_unique_products(self) -> pd.DataFrame:
        """
        Get unique products for vectorization.
        Returns a DataFrame with unique StockCode-Description combinations.
        """
        logger.info("Extracting unique products...")
        
        # Get unique products by StockCode
        unique_products = self.df.groupby("StockCode").agg({
            "Description": "first",
            "UnitPrice": "mean",  # Average price across transactions
            "Country": "first",  # Most common country
        }).reset_index()
        
        # Rename columns for clarity
        unique_products.columns = ["StockCode", "Description", "UnitPrice", "Country"]
        
        logger.info(f"Found {len(unique_products)} unique products")
        return unique_products
    
    def save(self) -> str:
        """Save the cleaned dataset to CSV."""
        logger.info(f"Saving cleaned data to {self.output_path}")
        self.df.to_csv(self.output_path, index=False)
        self.stats["cleaned_rows"] = len(self.df)
        logger.info(f"Saved {self.stats['cleaned_rows']} rows")
        return self.output_path
    
    def clean(self) -> Tuple[pd.DataFrame, dict]:
        """
        Run the full cleaning pipeline.
        
        Returns:
            Tuple of (cleaned DataFrame, statistics dict).
        """
        logger.info("Starting data cleaning pipeline...")
        
        self.load_data()
        
        # Chain all cleaning operations
        (self
            .clean_invoice_no()
            .clean_stock_code()
            .clean_description()
            .clean_quantity()
            .clean_unit_price()
            .clean_customer_id()
            .clean_country()
            .clean_invoice_date()
            .remove_duplicates()
            .handle_missing_values()
            .filter_valid_data()
        )
        
        self.save()
        
        logger.info("Data cleaning complete!")
        logger.info(f"Statistics: {self.stats}")
        
        return self.df, self.stats


if __name__ == "__main__":
    # Test the cleaner
    import sys
    sys.path.append("..")
    from config import config
    
    cleaner = DataCleaner(
        input_path=config.RAW_DATASET_PATH,
        output_path=config.CLEANED_DATASET_PATH
    )
    
    df, stats = cleaner.clean()
    print(f"\nCleaning Statistics:")
    print(f"  Original rows: {stats['original_rows']}")
    print(f"  Cleaned rows: {stats['cleaned_rows']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Nulls removed: {stats['nulls_removed']}")
    
    # Show unique products
    unique = cleaner.get_unique_products()
    print(f"\n  Unique products: {len(unique)}")
