"""
Data Cleaner for ProductLens AI.

Handles cleaning and preprocessing of e-commerce product datasets.
"""

import re
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from core.base_service import BaseService
from core.exceptions import ValidationError, DataError

logging.basicConfig(level=logging.INFO)


@dataclass
class CleaningConfig:
    """
    Configuration for data cleaning operations.
    
    Attributes:
        remove_cancelled: Remove cancelled orders (InvoiceNo starting with 'C').
        remove_negative_prices: Remove items with negative prices.
        min_description_length: Minimum description length to keep.
        max_description_length: Maximum description length to keep.
        fill_missing_country: Value for missing country.
        uppercase_descriptions: Convert descriptions to uppercase.
    """
    remove_cancelled: bool = True
    remove_negative_prices: bool = True
    min_description_length: int = 3
    max_description_length: int = 200
    fill_missing_country: str = "Unknown"
    uppercase_descriptions: bool = True
    
    # Noise patterns to remove
    noise_patterns: Dict[str, str] = field(default_factory=lambda: {
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
    })


@dataclass
class CleaningResult:
    """
    Results from the data cleaning process.
    
    Attributes:
        dataframe: Cleaned pandas DataFrame.
        original_rows: Number of rows before cleaning.
        cleaned_rows: Number of rows after cleaning.
        duplicates_removed: Number of duplicate rows removed.
        nulls_removed: Number of rows removed due to null values.
        invalid_removed: Number of invalid rows removed.
    """
    dataframe: pd.DataFrame
    original_rows: int
    cleaned_rows: int
    duplicates_removed: int
    nulls_removed: int
    invalid_removed: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding DataFrame)."""
        return {
            "original_rows": self.original_rows,
            "cleaned_rows": self.cleaned_rows,
            "duplicates_removed": self.duplicates_removed,
            "nulls_removed": self.nulls_removed,
            "invalid_removed": self.invalid_removed,
            "retention_rate": f"{self.cleaned_rows / self.original_rows * 100:.1f}%"
        }


class DataCleaner(BaseService):
    """
    Service for cleaning and preprocessing e-commerce datasets.
    
    Handles:
    - Removing special character noise
    - Handling missing values
    - Removing duplicates
    - Standardizing formats
    - Filtering invalid entries
    
    The cleaner uses a fluent interface pattern, allowing
    method chaining for customized cleaning pipelines.
    
    Example:
        >>> cleaner = DataCleaner()
        >>> result = cleaner.clean("raw_data.csv", "cleaned_data.csv")
        >>> print(result.to_dict())
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize the DataCleaner.
        
        Args:
            config: Cleaning configuration options.
        """
        super().__init__("DataCleaner")
        self.config = config or CleaningConfig()
        self._df: Optional[pd.DataFrame] = None
        self._stats = {
            "original_rows": 0,
            "cleaned_rows": 0,
            "duplicates_removed": 0,
            "nulls_removed": 0,
            "invalid_removed": 0
        }
        self._mark_initialized()
    
    def load_data(self, input_path: str) -> "DataCleaner":
        """
        Load dataset from CSV file.
        
        Args:
            input_path: Path to the CSV file.
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValidationError: If file doesn't exist or can't be read.
        """
        path = Path(input_path)
        if not path.exists():
            raise ValidationError("input_path", f"File not found: {input_path}")
        
        try:
            self._df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
            self._stats["original_rows"] = len(self._df)
            self.logger.info(f"Loaded {self._stats['original_rows']} rows from {input_path}")
            return self
        except Exception as e:
            raise DataError(f"Failed to load CSV: {e}")
    
    def load_dataframe(self, df: pd.DataFrame) -> "DataCleaner":
        """
        Load data from an existing DataFrame.
        
        Args:
            df: Pandas DataFrame to clean.
            
        Returns:
            Self for method chaining.
        """
        self._df = df.copy()
        self._stats["original_rows"] = len(self._df)
        self.logger.info(f"Loaded DataFrame with {len(self._df)} rows")
        return self
    
    def _clean_text(self, text: Any) -> str:
        """
        Clean a text value by removing noise characters.
        
        Args:
            text: Value to clean.
            
        Returns:
            Cleaned string.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        result = str(text)
        for noise, replacement in self.config.noise_patterns.items():
            result = result.replace(noise, replacement)
        
        result = re.sub(r"\s+", " ", result)
        return result.strip()
    
    def _clean_numeric(self, value: Any) -> Optional[float]:
        """
        Clean a numeric value.
        
        Args:
            value: Value to clean.
            
        Returns:
            Cleaned float or None.
        """
        if pd.isna(value):
            return None
        
        try:
            val_str = str(value)
            for noise in self.config.noise_patterns.keys():
                val_str = val_str.replace(noise, "")
            val_str = val_str.strip()
            
            if not val_str:
                return None
            
            return float(val_str)
        except (ValueError, TypeError):
            return None
    
    def clean_invoice_no(self) -> "DataCleaner":
        """Clean the InvoiceNo column."""
        if "InvoiceNo" in self._df.columns:
            self._df["InvoiceNo"] = self._df["InvoiceNo"].apply(self._clean_text)
            self.logger.debug("Cleaned InvoiceNo column")
        return self
    
    def clean_stock_code(self) -> "DataCleaner":
        """Clean the StockCode column."""
        if "StockCode" in self._df.columns:
            self._df["StockCode"] = self._df["StockCode"].apply(self._clean_text)
            self.logger.debug("Cleaned StockCode column")
        return self
    
    def clean_description(self) -> "DataCleaner":
        """Clean the Description column."""
        if "Description" in self._df.columns:
            self._df["Description"] = self._df["Description"].apply(self._clean_text)
            if self.config.uppercase_descriptions:
                self._df["Description"] = self._df["Description"].str.upper()
            self.logger.debug("Cleaned Description column")
        return self
    
    def clean_quantity(self) -> "DataCleaner":
        """Clean the Quantity column."""
        if "Quantity" in self._df.columns:
            self._df["Quantity"] = self._df["Quantity"].apply(self._clean_numeric)
            self._df["Quantity"] = pd.to_numeric(self._df["Quantity"], errors="coerce")
            self.logger.debug("Cleaned Quantity column")
        return self
    
    def clean_unit_price(self) -> "DataCleaner":
        """Clean the UnitPrice column."""
        if "UnitPrice" in self._df.columns:
            self._df["UnitPrice"] = self._df["UnitPrice"].apply(self._clean_numeric)
            self.logger.debug("Cleaned UnitPrice column")
        return self
    
    def clean_customer_id(self) -> "DataCleaner":
        """Clean the CustomerID column."""
        if "CustomerID" in self._df.columns:
            self._df["CustomerID"] = self._df["CustomerID"].apply(self._clean_numeric)
            self.logger.debug("Cleaned CustomerID column")
        return self
    
    def clean_country(self) -> "DataCleaner":
        """Clean the Country column."""
        if "Country" in self._df.columns:
            self._df["Country"] = self._df["Country"].apply(self._clean_text)
            self.logger.debug("Cleaned Country column")
        return self
    
    def clean_invoice_date(self) -> "DataCleaner":
        """Clean and parse the InvoiceDate column."""
        if "InvoiceDate" in self._df.columns:
            self._df["InvoiceDate"] = pd.to_datetime(
                self._df["InvoiceDate"],
                errors="coerce",
                format="%Y-%m-%d %H:%M:%S"
            )
            self.logger.debug("Cleaned InvoiceDate column")
        return self
    
    def remove_duplicates(self) -> "DataCleaner":
        """Remove duplicate rows."""
        before = len(self._df)
        self._df = self._df.drop_duplicates()
        self._stats["duplicates_removed"] = before - len(self._df)
        self.logger.info(f"Removed {self._stats['duplicates_removed']} duplicates")
        return self
    
    def handle_missing_values(self) -> "DataCleaner":
        """Handle missing values in critical columns."""
        before = len(self._df)
        
        # Drop rows with missing critical columns
        critical = ["StockCode", "Description"]
        existing_critical = [c for c in critical if c in self._df.columns]
        if existing_critical:
            self._df = self._df.dropna(subset=existing_critical)
        
        # Drop empty descriptions
        if "Description" in self._df.columns:
            self._df = self._df[self._df["Description"].str.len() > 0]
        
        # Fill missing values
        if "Country" in self._df.columns:
            self._df["Country"] = self._df["Country"].fillna(self.config.fill_missing_country)
        
        if "UnitPrice" in self._df.columns:
            self._df["UnitPrice"] = self._df["UnitPrice"].fillna(0.0)
        
        self._stats["nulls_removed"] = before - len(self._df)
        self.logger.info(f"Removed {self._stats['nulls_removed']} rows with missing values")
        return self
    
    def filter_valid_data(self) -> "DataCleaner":
        """Filter out invalid data entries."""
        before = len(self._df)
        
        # Remove negative prices
        if self.config.remove_negative_prices and "UnitPrice" in self._df.columns:
            self._df = self._df[self._df["UnitPrice"] >= 0]
        
        # Filter description length
        if "Description" in self._df.columns:
            self._df = self._df[
                (self._df["Description"].str.len() >= self.config.min_description_length) &
                (self._df["Description"].str.len() <= self.config.max_description_length)
            ]
        
        # Remove cancelled orders
        if self.config.remove_cancelled and "InvoiceNo" in self._df.columns:
            self._df = self._df[~self._df["InvoiceNo"].str.startswith("C", na=False)]
        
        self._stats["invalid_removed"] = before - len(self._df)
        self.logger.info(f"Filtered {self._stats['invalid_removed']} invalid rows")
        return self
    
    def get_unique_products(self) -> pd.DataFrame:
        """
        Get unique products for vectorization.
        
        Returns:
            DataFrame with unique StockCode-Description combinations.
        """
        self.logger.info("Extracting unique products...")
        
        unique = self._df.groupby("StockCode").agg({
            "Description": "first",
            "UnitPrice": "mean",
            "Country": "first"
        }).reset_index()
        
        unique.columns = ["StockCode", "Description", "UnitPrice", "Country"]
        self.logger.info(f"Found {len(unique)} unique products")
        return unique
    
    def save(self, output_path: str) -> str:
        """
        Save cleaned data to CSV.
        
        Args:
            output_path: Path for output file.
            
        Returns:
            Output path.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        self._df.to_csv(output, index=False)
        self._stats["cleaned_rows"] = len(self._df)
        self.logger.info(f"Saved {self._stats['cleaned_rows']} rows to {output_path}")
        return output_path
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the current DataFrame."""
        return self._df
    
    def clean(
        self,
        input_path: str,
        output_path: str
    ) -> CleaningResult:
        """
        Run the full cleaning pipeline.
        
        Args:
            input_path: Path to raw data CSV.
            output_path: Path for cleaned data CSV.
            
        Returns:
            CleaningResult with statistics.
        """
        self.logger.info("Starting data cleaning pipeline...")
        
        (self
            .load_data(input_path)
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
        
        self.save(output_path)
        
        self.logger.info("Data cleaning complete!")
        
        return CleaningResult(
            dataframe=self._df,
            original_rows=self._stats["original_rows"],
            cleaned_rows=self._stats["cleaned_rows"],
            duplicates_removed=self._stats["duplicates_removed"],
            nulls_removed=self._stats["nulls_removed"],
            invalid_removed=self._stats["invalid_removed"]
        )
    
    def health_check(self) -> dict:
        """Check service health."""
        base = super().health_check()
        base["pandas_available"] = True
        base["current_data_rows"] = len(self._df) if self._df is not None else 0
        return base
