# HW2
# Name: Yuhui Wang
# Collaborators:
# Date: 4/26/2024

import random

def count_characters(s):
    """Count the occurrences of each character in the input string.

    Args:
        s (str): The input string to count characters from.

    Returns:
        dict: A dictionary where keys are characters and values are their corresponding counts.
    """
    d = {}
    for i in range(len(s)):
        if s[i] not in d:
            d[s[i]] = 1
        else:
            d[s[i]] += 1
            
    return d


def count_ngrams(string, n=1):
    """Count the occurrences of each n-gram in the input string.

    Args:
        string (str): The input string to count n-grams from.
        n (int, optional): The order of the n-grams. Defaults to 1.

    Returns:
        dict: A dictionary where keys are n-grams and values are their corresponding counts.
    """
    counts = {}
    for i in range(len(string) - n + 1):
        ngram = string[i:i+n]
        if ngram in counts:
            counts[ngram] += 1
        else:
            counts[ngram] = 1
    return counts


def markov_text(s, n, length=100, seed=""):
    """Generate synthetic text based on an n-th order Markov model.

    Args:
        s (str): The input string to generate text from.
        n (int): The order of the Markov model.
        length (int, optional): The desired length of the generated text. Defaults to 100.
        seed (str, optional): The initial string to start the generation process. Defaults to an empty string.

    Returns:
        str: The generated synthetic text.
    """
    ngram_counts = count_ngrams(s, n+1)
    generated_text = seed

    while len(generated_text) < len(seed) + length:
        current_ngram = generated_text[-n:]
        possible_next_chars = [key[n] for key in ngram_counts if key.startswith(current_ngram)]
        weights = [ngram_counts[current_ngram + char] for char in possible_next_chars]
        next_char = random.choices(possible_next_chars, weights)[0]
        generated_text += next_char

    return generated_text
