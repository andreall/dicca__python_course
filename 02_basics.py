#!/usr/bin/env python
# coding: utf-8

#  # Python basics
# 
# ### Andrea Lira Loarca | andrea.lira.loarca@unige.it

# ## Built-In Types

# | Type        | Example        | Description                                                  |
# |-------------|----------------|--------------------------------------------------------------|
# | ``int``     | ``x = 1``      | integers (i.e., whole numbers)                               |
# | ``float``   | ``x = 1.0``    | floating-point numbers (i.e., real numbers)                  |
# | ``complex`` | ``x = 1 + 2j`` | Complex numbers (i.e., numbers with real and imaginary part) |
# | ``bool``    | ``x = True``   | Boolean: True/False values                                   |
# | ``str``     | ``x = 'abc'``  | String: characters or text                                   |
# | ``NoneType``| ``x = None``   | Special object indicating nulls                              |

# ## Operators

# | Operator     | Name           | Description                                            |
# |--------------|----------------|--------------------------------------------------------|
# | ``a + b``    | Addition       | Sum of ``a`` and ``b``                                 |
# | ``a - b``    | Subtraction    | Difference of ``a`` and ``b``                          |
# | ``a * b``    | Multiplication | Product of ``a`` and ``b``                             |
# | ``a / b``    | True division  | Quotient of ``a`` and ``b``                            |
# | ``a // b``   | Floor division | Quotient of ``a`` and ``b``, removing fractional parts |
# | ``a % b``    | Modulus        | Integer remainder after division of ``a`` by ``b``     |
# | ``a ** b``   | Exponentiation | ``a`` raised to the power of ``b``                     |
# | ``-a``       | Negation       | The negative of ``a``                                  |
# | ``+a``       | Unary plus     | ``a`` unchanged (rarely used)                          |

# In[1]:


# addition, subtraction, multiplication
print((4 + 8) * (6.5 - 3))


# In[2]:


# Division
print(11 / 2)


# In[3]:


3 / 2


# Note that in Python 2.x, this used to return ``1`` because it would round the solution to an integer. If you ever need to work with Python 2 code, the safest approach is to add the following line at the top of the script:
# 
#     from __future__ import division
#     
# and the division will then behave like a Python 3 division. Note that in Python 3 you can also specifically request integer division:

# In[4]:


3 // 2


# ### Assignment Operations

# In[5]:


a = 24
print(a)


# In[6]:


print(type(a))


# In[7]:


b = a + 3.5
print(b)
print(type(b))


# In[8]:


print(f'Variable {a} is {type(a)}')


# In[9]:


print('Variable {} is {}'.format(a, type(a)))


# In[10]:


a = a + 2
print(a)


# In[11]:


print(a + 2)


# In[12]:


a += 2  # equivalent to a = a + 2
print(a)


# ### Comparison Operations

# | Operation     | Description                       |
# |---------------|-----------------------------------|
# | ``a == b``    | ``a`` equal to ``b``              |
# | ``a < b``     | ``a`` less than ``b``             |
# | ``a <= b``    | ``a`` less than or equal to ``b`` |
# | ``a != b``    | ``a`` not equal to ``b``             |
# | ``a > b``     | ``a`` greater than ``b``             |
# | ``a >= b``    | ``a`` greater than or equal to ``b`` |

# In[13]:


print(22 / 2 == 10 + 1)


# In[14]:


# 25 is even
print(25 % 2 == 0)


# In[15]:


# 66 is odd
print(66 % 2 == 0)


# In[16]:


# check if a is between 15 and 30
a = 25
print(15 < a < 30)


# ### Boolean Operations

# In[17]:


x = 4
print((x < 6) and (x > 2))


# In[18]:


print((x > 10) or (x % 2 == 0))


# In[19]:


print(not (x < 6))


# ### Membership Operators

# | Operator      | Description                                       |
# |---------------|---------------------------------------------------|
# | ``a in b``    | True if ``a`` is a member of ``b``                |
# | ``a not in b``| True if ``a`` is not a member of ``b``            |

# In[20]:


print(1 in [1, 2, 3])


# In[21]:


print(2 not in [1, 2, 3])


# ### Floating-Point Numbers

# In[22]:


x = 0.000005
y = 5e-6
print(x == y)


# In[23]:


print(0.1 + 0.2 == 0.3)


# > Floating-point precision is limited, which can cause equality tests to be unstable

# ### String Type

# <center> <img src="img/list-indexing.png" width="1600"/> </center>

# In[24]:


message = "what do you like?"
print(message)


# You can use either single quotes (``'``), double quotes (``"``), or triple quotes (``'''`` or ``"""``) to enclose a string (the last one is used for multi-line strings). To include single or double quotes inside a string, you can either use the opposite quote to enclose the string:

# In[25]:


response = "I'm"
print(response)


# In[26]:


response = '"hello"'
print(response)


# In[27]:


# length of string
print(len(response))


# In[28]:


# Make upper-case. See also str.lower()
print(response.upper())


# In[29]:


# Capitalize. See also str.title()
print(message.capitalize())


# In[30]:


s = "Spam egg spam spam"
s.index('egg ')  # An integer giving the position of the sub-string


# In[31]:


s.split()


# In[32]:


s2 = "Spam-egg-spam_spam"
s2.split()


# In[33]:


s3 = s2.split('-')
s3


# In[34]:


' '.join(s3)


# In[35]:


# concatenation with +
print(message + response)


# In[36]:


# concatenation with +
print(f'{message} {response}')


# In[37]:


# concatenation with +
print('{0} {1}'.format(message, response))
print('{1} -- {0} {1}'.format(message, response))


# In[38]:


# multiplication is multiple concatenation
print(5 * response)


# In[39]:


# Access individual characters (zero-based indexing)
print(message[0])


# In[40]:


cadena1 = "cool"
print(cadena1[::-1])


# ## Exercise 2
# Given a string such as the one below, make a new string that does not contain the word ``egg``:

# In[41]:


a = "Hello, egg world!"

# enter your solution here


# ### Boolean Type

# In[42]:


result = 4 < 5
print(result)


# > ``True`` and ``False`` must be capitalized!

# ## Built-In Data Structures

# | Type Name | Example                   |Description                            |
# |-----------|---------------------------|---------------------------------------|
# | ``list``  | ``[1, 2, 3]``             | Ordered collection                    |
# | ``tuple`` | ``(1, 2, 3)``             | Immutable ordered collection          |
# | ``dict``  | ``{'a': 1, 'b': 2, 'c': 3}`` | (key,value) mapping                |

# ### Lists

# In[43]:


lst = [2, 3, 5, 7]


# In[44]:


li = [4, 5.5, "spam"]


# In[45]:


# Length of a list
print(len(lst))


# In[46]:


# Change value of list
li[1] = -2.2


# In[47]:


# Append a value to the end
lst.append(11)
print(lst)


# In[48]:


# Addition concatenates lists
print(lst + [13, 17, 19])


# In[49]:


# sort() method sorts in-place
lst = [2, 5, 1, 6, 3, 4]
lst.sort()
print(lst)


# In[50]:


lst = [1, "two", 3.14, [0, 3, 5]]
print(lst)


# #### List indexing and slicing

# In[51]:


lst = [2, 3, 5, 7, 11]


# In[52]:


print(lst[0])


# In[53]:


print(lst[1])


# In[54]:


print(lst[-1])


# In[55]:


print(lst[0:3])


# In[56]:


print(lst[:3])


# In[57]:


print(lst[::2])  # equivalent to l[0:len(l):2]


# In[58]:


lst[0] = 100
print(lst)


# ### Dictionaries

# In[59]:


numbers = {"one": 1, "two": 2, "three": 3}


# In[60]:


# Access a value via the key
print(numbers["two"])


# In[61]:


# Set a new key:value pair
numbers["ninety"] = 90
print(numbers)


# ### Exercises

# - Create a dictionary with the following birthday information:
# 
# > - Albert Einstein - 03/14/1879
# > - Benjamin Franklin - 01/17/1706
# > - Ada Lovelace - 12/10/1815
# > - Marie Curie - 07/11/1867
# > - Rowan Atkinson - 01/6/1955
# > - Rosalind Franklin - 25/07/1920

# - Check if Marie Curie is in our dictonary
# - Get Albert Einstein's birthday

# In[62]:


# Create a dictionary with the following birthday information


# In[63]:


birthdays = {
    "Albert Einstein": "03/14/1879",
    "Benjamin Franklin": "01/17/1706",
    "Ada Lovelace": "12/10/1815",
    "Marie Curie": "07/11/1867",
    "Rowan Atkinson": "01/6/1955",
    "Rosalind Franklin": "25/07/1920",
}


# In[64]:


# Check if Marie Curie is in our dictonary


# In[65]:


print("Marie Curie" in birthdays)


# In[66]:


# Get Albert Einstein's birthday


# In[67]:


print(birthdays["Albert Einstein"])


# ## Control Flow

# ### Conditional Statements: ``if``-``elif``-``else``

# In[68]:


x = 15

if x == 0:
    print(x, "is zero")
elif (x > 0) and (x < 10):
    print(x, "is between 0 and 10")
elif (x > 10) and (x < 20):
    print(x, "is between 10 and 20")
else:
    print(x, "is negative")
print("Hello world")


# ### Loops: ``for``

# In[69]:


n = [2, 3, 5, 7]
for e in n:
    print(e)


# > Please avoid Matlab-like for statements with range

# In[70]:


for e in range(len(n)):
    print(n[e])


# In[71]:


m = range(5)  # m = [0, 1, 2, 3, 4]
for _ in m:
    print("hola")


# In[72]:


for index, value in enumerate(n):
    print("The value of index ", index, " is ", value)


# ### Loops: ``while``

# In[73]:


i = 0
while i < 10:
    print(i)
    i += 1


# > Be careful with while loops

# ## Functions

# ### Defining Functions

# In[74]:


import time


def header():
    text = "This is a function"
    text += ". Copyright " + time.strftime("%d-%m-%Y")

    return text


print(header())


# In[75]:


def header_v2(author):
    text = "This function is written by "
    text += author
    text += f". Copyright {time.strftime('%d-%m-%Y')}"

    return text


print(header_v2("Andrea Lira Loarca"))


# > Atenttion to the string format

# In[76]:


print(header_v2("Gabriella Ruffini"))


# In[77]:


def fibonacci(n):
    l = []
    a = 0
    b = 1
    while len(l) < n:
        l.append(a)
        c = a + b
        a = b
        b = c
    return l


# In[78]:


print(fibonacci(10))


# In[79]:


print(fibonacci(3))


# ### Default Argument Values

# In[80]:


def fibonacci(n, start=0):
    fib = []
    a = 0
    b = 1
    while len(fib) < n:
        if a >= start:
            fib.append(a)
        c = a + b
        a = b
        b = c
    return fib


# In[81]:


print(fibonacci(10))


# In[82]:


print(fibonacci(10, 5))


# In[83]:


## Keyword arguments


# In[84]:


print(fibonacci(start=5, n=10))


# In[85]:


from datetime import datetime, timedelta

dt1 = datetime(2005, 7, 14, 12, 30)

dt2 = dt1 + timedelta(hours=5)

print(dt2)


# **timedelta**(`[days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]]`)
# 
# > All arguments are optional and default to 0. Arguments may be ints, longs, or floats, and may be positive or negative.

# ### Documentation strings (docstrings)

# * Python documentation strings (docstrings) provide a convenient **way of associating documentation with Python functions** and modules.
# * Docstrings can be written following **several styles**. We use [Google Python Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/).
# * An object's docsting is defined by including a **string constant as the first statement in the function's definition**.

# * Unlike conventional source code comments **the docstring should describe what the function does, not how**.
# * **All functions should have a docstring**.
# * This allows to inspect these comments at run time, for instance as an **interactive help system**, or **export them as HTML, LaTeX, PDF** or other formats.

# In[86]:


def fibonacci(n, start=0):
    """Build a Fibonacci series with n elements starting at start

    Args:
        n: number of elements
        start: lower limit. Default 0

    Returns:
        A list with a Fibonacci series with n elements
    """
    fib = []
    a = 0
    b = 1
    while len(fib) < n:
        if a >= start:
            fib.append(a)
        c = a + b
        a = b
        b = c
    return fib


# ## Modules and Packages

# ### Loading Modules: the ``import`` Statement

# #### Explicit module import by alias

# In[87]:


import numpy as np


# #### Explicit import of module contents

# In[88]:


from scipy.stats import norm


# ### Importing from Third-Party Modules

# The best way to import libraries is included in their official help
# 
# ```python
# import math
# import numpy as np
# from scipy import linalg, optimize
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import sympy
# ```
