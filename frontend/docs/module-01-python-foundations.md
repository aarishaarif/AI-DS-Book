---
title: Python Foundations
sidebar_position: 0
---

# Module 1: Python Foundations

Welcome to Python Foundations! This module will teach you the essential programming concepts you need to succeed in AI and Data Science.

## Learning Objectives

By the end of this module, you will be able to:

- Understand Python syntax and data structures
- Write functions and control flow
- Read and write files
- Use common libraries for data manipulation

## Prerequisites

No prior programming experience required. All necessary concepts will be explained from the basics.

## 1.1 Variables and Data Types

Python is dynamically typed, which means you don't need to specify the type of a variable when you create it. The type is determined by the value you assign.

### Example 1: Numeric Types

```python
# Integers (whole numbers)
age = 25
print(f"Age: {age}")  # Age: 25

# Floats (decimal numbers)
pi = 3.14159
temperature = 98.6
print(f"Temperature: {temperature}°F")  # Temperature: 98.6°F

# Booleans
is_student = True
is_professor = False
print(f"Student: {is_student}, Professor: {is_professor}")  # Student: True, Professor: False
```

### Example 2: String Types

```python
# Strings in single or double quotes
name = "Alice"
class_name = 'AI 101'
message = f"Hello, {name}! Welcome to {class_name}."
print(message)  # Hello, Alice! Welcome to AI 101.

# Multi-line strings (useful for documentation)
docstring = """
This is a multi-line string.
It preserves line breaks and spaces.
Great for writing descriptions!
"""
print(docstring)
```

### Example 3: List and Tuple

```python
# Lists (mutable, ordered collections)
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = ["text", 123, 3.14, True]

print(f"First fruit: {fruits[0]}")  # First fruit: apple
print(f"Length of fruits: {len(fruits)}")  # Length of fruits: 3

# Lists are mutable - you can modify them
fruits.append("orange")  # Adds "orange" to the end
print(fruits)  # ["apple", "banana", "cherry", "orange"]

# Tuples (immutable, ordered collections)
coordinates = (10, 20)
temp_range = (15.0, 25.0, 30.0)

print(f"X coordinate: {coordinates[0]}")  # X coordinate: 10

# Tuples are immutable - this would raise an error:
# coordinates[0] = 5  # TypeError: 'tuple' object does not support item assignment
```

## 1.2 Control Flow

Control flow allows your program to make decisions and repeat actions.

### Example 1: If-Else Statements

```python
age = 18

if age >= 18:
    print("You are an adult.")
elif age >= 13:
    print("You are a teenager.")
else:
    print("You are a child.")

# Note: In Python, indentation matters!
if age >= 18:
    print("You can vote.")
    if age >= 21:
        print("You can drink alcohol.")
else:
    print("You cannot vote yet.")
```

### Example 2: For Loops

```python
# Iterating over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"I like {fruit}")

# Iterating over a range
for i in range(5):
    print(f"Count: {i}")  # Count: 0, 1, 2, 3, 4

# Iterating with enumeration
for index, fruit in enumerate(fruits):
    print(f"{index + 1}. {fruit}")  # 1. apple, 2. banana, 3. cherry
```

### Example 3: While Loops

```python
# Basic while loop
count = 0
while count < 3:
    print(f"Count: {count}")
    count += 1  # or count = count + 1

# Using while with user input
answer = ""
while answer != "quit":
    answer = input("Type 'quit' to exit: ")
    if answer != "quit":
        print(f"You typed: {answer}")
```

## 1.3 Functions

Functions are reusable blocks of code that perform specific tasks.

### Example 1: Basic Function

```python
def greet(name, greeting="Hello"):
    """Greet a person with an optional greeting."""
    message = f"{greeting}, {name}!"
    print(message)
    return message

# Calling the function
greet("Alice")  # Output: Hello, Alice!
greet("Bob", "Hi")  # Output: Hi, Bob!
```

### Example 2: Function with Return Value

```python
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    
    total = 0
    for num in numbers:
        total += num
    
    average = total / len(numbers)
    return average

# Example usage
scores = [85, 92, 78, 96, 88]
avg_score = calculate_average(scores)
print(f"Average score: {avg_score}")  # Average score: 87.8
```

### Example 3: Lambda Functions (Anonymous Functions)

```python
# Lambda function for simple operations
square = lambda x: x ** 2
print(square(5))  # 25

double = lambda x: x * 2
print(double(21))  # 42

# Using lambda with map and filter
numbers = [1, 2, 3, 4, 5]

# Apply square function to each number
squared = list(map(lambda x: x ** 2, numbers))
print(f"Squared: {squared}")  # Squared: [1, 4, 9, 16, 25]

# Filter even numbers
odd_numbers = list(filter(lambda x: x % 2 == 1, numbers))
print(f"Odd numbers: {odd_numbers}")  # Odd numbers: [1, 3, 5]
```

## 1.4 File Input/Output

Reading and writing files is essential for data processing.

### Example 1: Reading Files

```python
# Reading a file using 'with' statement (recommended)
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

# Reading file line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())  # strip() removes the newline character
```

### Example 2: Writing Files

```python
# Writing a new file (overwrites existing content)
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a test file.\n")

# Appending to an existing file
with open("output.txt", "a") as file:
    file.write("Appending this line.\n")

# Using with pandas for data analysis
import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Tokyo']
}

# Write to CSV
pd.DataFrame(data).to_csv('people.csv', index=False)
print("Data written to people.csv")
```

## 1.5 Common Python Libraries

### NumPy for Numerical Computing

```python
import numpy as np

# Creating arrays
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Basic operations
sum_array = np.sum(array_1d)  # 15
mean_array = np.mean(array_1d)  # 3.0

print(f"Array sum: {sum_array}")
print(f"Array mean: {mean_array}")
```

### Collections Module

```python
import collections

# Using defaultdict (provides default values for missing keys)
word_count = collections.defaultdict(int)
words = ["apple", "banana", "apple", "orange", "banana", "apple"]

for word in words:
    word_count[word] += 1

print(word_count)  # defaultdict(<class 'int'>, {'apple': 3, 'banana': 2, 'orange': 1})

# Using Counter (efficient for counting)
word_freq = collections.Counter(words)
print(f"Most common: {word_freq.most_common(1)}")  # [('apple', 3)]
```

## 1.6 Error Handling

Good code anticipates and handles errors gracefully.

### Example 1: Try-Except

```python
try:
    # This will raise an error
    result = 10 / 0
    print(f"Result: {result}")
except ZeroDivisionError:
    print("Error: Cannot divide by zero!")

# Try-except-finally
try:
    with open("file.txt", "r") as file:
        content = file.read()
    print("File read successfully")
except FileNotFoundError:
    print("Error: File not found")
except IOError:
    print("Error: Could not read the file")
finally:
    print("Operation completed (cleanup in finally block)")
```

## 1.7 Summary

In this module, you learned:

1. **Variables and Data Types**: Integers, floats, booleans, strings, lists, tuples
2. **Control Flow**: if-elif-else, for loops, while loops
3. **Functions**: Basic functions, parameters, return values, lambda functions
4. **File I/O**: Reading and writing files
5. **Common Libraries**: NumPy, collections module
6. **Error Handling**: try-except blocks

## 1.8 Practice Questions

1. Write a function to calculate the factorial of a number.
2. Create a program that reads a text file and counts word frequencies.
3. Explain the difference between lists and tuples.
4. Write a function that takes a list and returns a new list with only even numbers.
5. What happens when you try to access an index that doesn't exist in a list?

## 1.9 Further Reading

- [Python Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Project: Your First Python Script

Create a Python script that:
1. Takes a list of numbers as input
2. Calculates the mean, median, and mode
3. Writes the results to a CSV file
4. Creates a simple bar chart of the data

Save this script as `data_analysis.py` and run it to see your results!