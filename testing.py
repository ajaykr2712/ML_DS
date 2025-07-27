str = "CarccrMonk Publications."
print("".join(reversed(str)))

list = [1, 2, 3, 4, 5]
def reverse_list(lst):
    return lst[::-1]
print(reverse_list(list))
def reverse_string(s):
    return s[::-1]
print(reverse_string(str))


array = [1,2,4,5]
print(array[::-1])


dict = {"name": "Alice", "age": 30, "city": "New York"}
print(dict["city"][::-1])  # Reverse the string value of the "age" key


def reverse_words_in_sentence(s):
    """
    Reverses the order of words in a sentence.
    Example: "Hello world" -> "world Hello"
    """
    words = s.strip().split()
    return " ".join(reversed(words))

def my_language(s):
    cookbook = s.strip().split()
    return " ".join(reversed(cookbook))
print(my_language("Hello world"))