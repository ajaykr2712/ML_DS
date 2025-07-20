import pytest
from src.script import add_numbers, multiply_numbers, mean_of_list

def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0

def test_multiply_numbers():
    assert multiply_numbers(2, 3) == 6
    assert multiply_numbers(-1, 1) == -1
    assert multiply_numbers(0, 5) == 0

def test_mean_of_list():
    assert mean_of_list([1, 2, 3, 4, 5]) == 3.0
    assert mean_of_list([10, 20, 30]) == 20.0
    with pytest.raises(ValueError):
        mean_of_list([])

if __name__ == "__main__":
    pytest.main()
