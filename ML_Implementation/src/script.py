import numpy as np

def add_numbers(a: float, b: float) -> float:
    """Returns the sum of two numbers."""
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    """Returns the product of two numbers."""
    return a * b

def mean_of_list(numbers: list) -> float:
    """Returns the mean of a list of numbers."""
    if not numbers:
        raise ValueError("List cannot be empty")
    return np.mean(numbers)

if __name__ == "__main__":
    print("Sum:", add_numbers(3, 5))
    print("Product:", multiply_numbers(3, 5))
    print("Mean:", mean_of_list([1, 2, 3, 4, 5]))
