"""
Unit Tests for DataCleaner.

Tests the data cleaning pipeline for product descriptions.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from io import StringIO

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestDataCleaner:
    """Tests for DataCleaner."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a DataCleaner instance."""
        from ml.data.data_cleaner import DataCleaner
        return DataCleaner(
            min_desc_length=10,
            max_desc_length=200,
            min_word_count=2
        )
    
    @pytest.fixture
    def sample_data(self):
        """Sample product data for testing."""
        return pd.DataFrame({
            "StockCode": ["22384", "22423", "TEST", "A"],
            "Description": [
                "LUNCH BAG PINK POLKADOT",
                "REGENCY CAKESTAND 3 TIER",
                "a",  # Too short
                None  # Null value
            ]
        })
    
    def test_clean_description_valid(self, cleaner):
        """Test cleaning a valid description."""
        desc = "LUNCH BAG PINK POLKADOT"
        result = cleaner._clean_description(desc)
        
        assert result is not None
        assert result == "LUNCH BAG PINK POLKADOT"
    
    def test_clean_description_lowercase(self, cleaner):
        """Test that descriptions are uppercased."""
        desc = "lunch bag pink polkadot"
        result = cleaner._clean_description(desc)
        
        assert result == "LUNCH BAG PINK POLKADOT"
    
    def test_clean_description_whitespace(self, cleaner):
        """Test that extra whitespace is normalized."""
        desc = "  LUNCH    BAG   POLKADOT  "
        result = cleaner._clean_description(desc)
        
        assert "  " not in result
        assert result.strip() == result
    
    def test_clean_description_too_short(self, cleaner):
        """Test that short descriptions are rejected."""
        result = cleaner._clean_description("AB")
        assert result is None
    
    def test_clean_description_too_long(self, cleaner):
        """Test that long descriptions are rejected."""
        long_desc = "A" * 250
        result = cleaner._clean_description(long_desc)
        assert result is None
    
    def test_clean_description_null(self, cleaner):
        """Test that null descriptions are rejected."""
        result = cleaner._clean_description(None)
        assert result is None
    
    def test_is_valid_stock_code(self, cleaner):
        """Test stock code validation."""
        assert cleaner._is_valid_stock_code("22384") is True
        assert cleaner._is_valid_stock_code("A123") is True
        assert cleaner._is_valid_stock_code("") is False
        assert cleaner._is_valid_stock_code(None) is False
    
    def test_is_garbage_text(self, cleaner):
        """Test garbage text detection."""
        assert cleaner._is_garbage_text("TEST") is True
        assert cleaner._is_garbage_text("SAMPLE") is True
        assert cleaner._is_garbage_text("LUNCH BAG") is False
        assert cleaner._is_garbage_text("???") is True
    
    def test_remove_special_characters(self, cleaner):
        """Test special character removal."""
        desc = "LUNCH BAG™ @2024"
        result = cleaner._remove_special_characters(desc)
        
        assert "™" not in result
        assert "@" not in result
    
    @pytest.mark.ml
    def test_clean_dataframe(self, cleaner, sample_data):
        """Test cleaning a dataframe."""
        result = cleaner.clean_dataframe(sample_data)
        
        # Should have valid rows only
        assert len(result) == 2  # Only valid descriptions
        assert "22384" in result["StockCode"].values
        assert "22423" in result["StockCode"].values
    
    @pytest.mark.ml
    def test_deduplicate_products(self, cleaner):
        """Test product deduplication."""
        data = pd.DataFrame({
            "StockCode": ["22384", "22384", "22423"],
            "Description": [
                "LUNCH BAG PINK",
                "LUNCH BAG PINK POLKADOT",  # Longer, should be kept
                "CAKESTAND"
            ]
        })
        
        result = cleaner._deduplicate_products(data)
        
        # Should have only unique stock codes
        assert len(result) == 2
        # Should keep the longer description
        matching = result[result["StockCode"] == "22384"]
        assert "POLKADOT" in matching["Description"].values[0]
    
    @pytest.mark.ml
    def test_cleaning_result(self, cleaner, sample_data):
        """Test that cleaning returns CleaningResult."""
        from ml.data.data_cleaner import CleaningResult
        
        result = cleaner.clean(sample_data)
        
        assert isinstance(result, CleaningResult)
        assert result.total_products > 0
        assert result.valid_products > 0
        assert result.removed_count >= 0


class TestDataCleanerConfiguration:
    """Tests for DataCleaner configuration."""
    
    def test_custom_length_limits(self):
        """Test custom length configuration."""
        from ml.data.data_cleaner import DataCleaner
        
        cleaner = DataCleaner(
            min_desc_length=5,
            max_desc_length=50
        )
        
        assert cleaner.min_desc_length == 5
        assert cleaner.max_desc_length == 50
    
    def test_custom_word_count(self):
        """Test custom word count configuration."""
        from ml.data.data_cleaner import DataCleaner
        
        cleaner = DataCleaner(min_word_count=3)
        
        # Two-word description should be rejected
        result = cleaner._clean_description("LUNCH BAG")
        assert result is None
        
        # Three-word description should pass
        result = cleaner._clean_description("LUNCH BAG PINK")
        assert result is not None
