"""
Comprehensive Test Suite for ML Implementation
=============================================

Advanced test cases for machine learning algorithms and utilities.

Author: ML Arsenal Team
Date: July 2025
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from script import add_numbers, multiply_numbers, mean_of_list
except ImportError:
    # Fallback for testing environment
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.script import add_numbers, multiply_numbers, mean_of_list


class TestBasicFunctions:
    """Test suite for basic utility functions."""
    
    def test_add_numbers_positive(self):
        """Test addition with positive numbers."""
        result = add_numbers(3, 5)
        assert result == 8
        assert isinstance(result, (int, float))
    
    def test_add_numbers_negative(self):
        """Test addition with negative numbers."""
        result = add_numbers(-3, -5)
        assert result == -8
    
    def test_add_numbers_mixed(self):
        """Test addition with mixed positive/negative numbers."""
        result = add_numbers(-3, 5)
        assert result == 2
    
    def test_add_numbers_zero(self):
        """Test addition with zero."""
        result = add_numbers(0, 5)
        assert result == 5
        result = add_numbers(5, 0)
        assert result == 5
    
    def test_add_numbers_floats(self):
        """Test addition with floating point numbers."""
        result = add_numbers(3.5, 2.5)
        assert abs(result - 6.0) < 1e-10
    
    def test_multiply_numbers_positive(self):
        """Test multiplication with positive numbers."""
        result = multiply_numbers(3, 5)
        assert result == 15
    
    def test_multiply_numbers_negative(self):
        """Test multiplication with negative numbers."""
        result = multiply_numbers(-3, -5)
        assert result == 15
    
    def test_multiply_numbers_mixed(self):
        """Test multiplication with mixed positive/negative numbers."""
        result = multiply_numbers(-3, 5)
        assert result == -15
    
    def test_multiply_numbers_zero(self):
        """Test multiplication with zero."""
        result = multiply_numbers(0, 5)
        assert result == 0
        result = multiply_numbers(5, 0)
        assert result == 0
    
    def test_multiply_numbers_floats(self):
        """Test multiplication with floating point numbers."""
        result = multiply_numbers(3.5, 2.0)
        assert abs(result - 7.0) < 1e-10
    
    def test_mean_of_list_integers(self):
        """Test mean calculation with integers."""
        result = mean_of_list([1, 2, 3, 4, 5])
        expected = 3.0
        assert abs(result - expected) < 1e-10
    
    def test_mean_of_list_floats(self):
        """Test mean calculation with floats."""
        result = mean_of_list([1.5, 2.5, 3.5])
        expected = 2.5
        assert abs(result - expected) < 1e-10
    
    def test_mean_of_list_single_element(self):
        """Test mean calculation with single element."""
        result = mean_of_list([42])
        assert result == 42
    
    def test_mean_of_list_empty_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="List cannot be empty"):
            mean_of_list([])
    
    def test_mean_of_list_negative_numbers(self):
        """Test mean calculation with negative numbers."""
        result = mean_of_list([-1, -2, -3])
        expected = -2.0
        assert abs(result - expected) < 1e-10


class TestAdvancedScenarios:
    """Test suite for advanced scenarios and edge cases."""
    
    def test_numpy_integration(self):
        """Test numpy integration and operations."""
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        assert result == 3.0
        
        # Test matrix operations
        matrix = np.array([[1, 2], [3, 4]])
        result = np.linalg.det(matrix)
        assert abs(result - (-2.0)) < 1e-10
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test division by zero handling
        with pytest.raises(ZeroDivisionError):
            1 / 0  # This should raise ZeroDivisionError
        
        # Test invalid array operations
        with pytest.raises(ValueError):
            arr = np.array([])
            np.mean(arr)  # This should raise ValueError
    
    @pytest.mark.parametrize("x,y,expected", [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (100, -50, 50),
        (0.5, 0.5, 1.0)
    ])
    def test_add_numbers_parametrized(self, x, y, expected):
        """Test add_numbers function with multiple parameter sets."""
        result = add_numbers(x, y)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.parametrize("numbers,expected", [
        ([1, 2, 3], 2.0),
        ([0], 0.0),
        ([1, 1, 1], 1.0),
        ([-1, 0, 1], 0.0),
        ([10, 20, 30], 20.0)
    ])
    def test_mean_of_list_parametrized(self, numbers, expected):
        """Test mean_of_list function with multiple parameter sets."""
        result = mean_of_list(numbers)
        assert abs(result - expected) < 1e-10


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
