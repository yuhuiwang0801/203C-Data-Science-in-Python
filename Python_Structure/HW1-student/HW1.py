# PIC 16A HW1
# Name: Yuhui Wang
# Collaborators:
# Date:4/16/2024

import random # This is only needed in Problem 5

# Problem 1

def print_s(s):
    ''' Prints a given string.
    Args:
        s: A string.
    Returns:
        None
    '''
    print(s)

# you do not have to add docstrings for the rest of these print_s_* functions.

def print_s_lines(s):
    lines = s.split('\n')
    
    # Iterate through each line
    for line in lines:
        parts = line.split(':')
        
        for part in parts:
            print(part.strip())

def print_s_parts(s):
    lines = s.split('\n')
    
    for line in lines:
        parts = line.split(':')
        
        for part in parts:
            p = part.strip()
            if p[-1] == 'd' and p[-2] == 'e':
                print(p)


def print_s_some(s):
    lines = s.split('\n')
    max_len = min(lines)
    for line in lines:
        if line != max_len:
            print(line)

def print_s_change(s):
    s = s.replace("math", "data science")
    s = s.replace("long division", "machine learning")
    print(s)

# Problem 2 

def make_count_dictionary(L):
    ''' WRITE YOUR OWN DOCSTRING HERE
    Create a dictionary counting the occurrences of each element in a given list.

    Args:
        L (list): A list of elements (can be of any type).

    Returns:
        None: Prints the dictionary where each key is an element from the list and the value is its count.
    '''
    #construct a empty dictionary
    d = {}
    #loop, if not in dict, give it equals to 1; if in dict, add 1
    for i in L:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return(d)


# Problem 3

def gimme_an_odd_number():
    ''' WRITE YOUR OWN DOCSTRING HERE
    Repeatedly prompts the user to enter an integer until an odd number is entered. 
    Collects all entered numbers in a list and prints them.

    Returns:
        None: All entered integers are collected in a list which is printed when an odd integer is entered.
    '''
    l = []
    while True:
        num = int(input("Please enter an integer."))
        l.append(num)
        if num % 2 == 1:
            break
        else:
            continue
    
    print(l)

# Problem 4

def get_triangular_numbers(k):
    ''' WRITE YOUR OWN DOCSTRING HERE
    Generates a list of the first 'k' triangular numbers.

    Args:
        k (int): The number of triangular numbers to generate.

    Returns:
        list: A list containing the first 'k' triangular numbers.
    '''
    l = []
    for i in range(1, k+1):
        n = 0
        for j in range(1, i+1):
            n += j
        l.append(n)
    return l


def get_consonants(s):
    ''' WRITE YOUR OWN DOCSTRING HERE
    Extracts all consonants from a given string and returns them as a list.

    Args:
        s (str): The string from which consonants are to be extracted.

    Returns:
        list: A list containing all consonants found in the string.
    '''
    l = []
    constraints = ["a", "e", "i", "o", "u", " ", ",", "."]
    for i in s:
        if i not in constraints:
            l.append(i)

    return l


def get_list_of_powers(X, k):
    ''' WRITE YOUR OWN DOCSTRING HERE
    Generates a list where each element is a list of the powers of an element from X up to the 'k'th power.

    Args:
        X (list): A list of numbers.
        k (int): The highest power to which each number in X is raised.

    Returns:
        list: A nested list where each sublist contains powers of an element from X up to the 'k'th power.
    '''
    l = []
    for i in X:
        t = []
        for j in range(0, k + 1):
            t.append(i ** j)
        l.append(t)
    return l


def get_list_of_even_powers(L, k):
    ''' WRITE YOUR OWN DOCSTRING HERE
    Generates a list where each element is a list of even powers of an element from L up to the 'k'th power.

    Args:
        L (list): A list of numbers.
        k (int): The highest power to consider.

    Returns:
        list: A nested list where each sublist contains even powers of an element from L up to the 'k'th power.
    '''
    l = []
    for i in L:
        t = []
        for j in range(0, k+1):
            if j % 2 == 0:
                t.append(i ** j)
        l.append(t)
    return l



# Problem 5

def random_walk(ub, lb):
    ''' WRITE YOUR OWN DOCSTRING HERE
    Simulates a random walk starting from position 0, where each step is either +1 or -1 (chosen randomly),
    until the position hits or exceeds an upper or lower boundary.

    Args:
        ub (int): The upper boundary of the walk.
        lb (int): The lower boundary of the walk.

    Returns:
        tuple: A tuple containing the final position, the list of positions (excluding the final position),
               and the list of steps taken (+1 or -1).
    '''
    # Initialize variables
    pos = 0
    positions = [pos]
    steps = []

    # Continue the walk until upper or lower bound is reached
    while True:
        # Flip a fair coin (heads = 1, tails = -1)
        step = random.choice([-1, 1])
        steps.append(step)
        
        # Update the position based on the coin flip
        pos += step
        positions.append(pos)

        # Check if upper or lower bound is reached
        if pos >= ub:
            print(f"Upper bound at {ub} reached.")
            break
        elif pos <= lb:
            print(f"Lower bound at {lb} reached.")
            break

    return pos, positions[:-1], steps


# If you uncomment these two lines, you can run 
# the gimme_an_odd_number() function by
# running this script on your IDE or terminal. 
# Of course you can run the function in notebook as well. 
# Make sure this stays commented when you submit
# your code.
#
# if __name__ == "__main__":
#     gimme_an_odd_number()