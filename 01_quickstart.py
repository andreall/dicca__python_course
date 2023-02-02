#!/usr/bin/env python
# coding: utf-8

# # Quickstart: zero to Python
# 
# Taken and modified from https://foundations.projectpythia.org/foundations/getting-started-python.html

# ## Interactively (demo)

# To run Python code interactively, one can use the standard Python prompt, which can be launched by typing ``python`` in your standard shell:
# 
#     $ python
#     Python 3.4.1 (default, May 21 2014, 21:17:51) 
#     [GCC 4.2.1 Compatible Apple Clang 4.1 ((tags/Apple/clang-421.11.66))] on darwin
#     Type "help", "copyright", "credits" or "license" for more information.
#     >>>
# 
# The ``>>>`` indicates that Python is ready to accept commands. If you type ``a = 1`` then press enter, this will assign the value ``1`` to ``a``. If you then type ``a`` you will see the value of ``a`` (this is equivalent to ``print a``):
# 
#     >>> a = 1
#     >>> a
#     1
# 
# The Python shell can execute any Python code, even multi-line statements, though it is often more convenient to use Python non-interactively for such cases.
# 
# The default Python shell is limited, and in practice, you will want instead to use the IPython (or interactive Python) shell. This is an add-on package that adds many features to the default Python shell, including the ability to edit and navigate the history of previous commands, as well as the ability to tab-complete variable and function names. To start up IPython, type:
# 
#     $ ipython
#     Python 3.4.1 (default, May 21 2014, 21:17:51) 
#     Type "copyright", "credits" or "license" for more information.
# 
#     IPython 2.1.0 -- An enhanced Interactive Python.
#     ?         -> Introduction and overview of IPython's features.
#     %quickref -> Quick reference.
#     help      -> Python's own help system.
#     object?   -> Details about 'object', use 'object??' for extra details.
# 
#     In [1]:
# 
# The first time you start up IPython, it will display a message which you can skip over by pressing ``ENTER``. The ``>>>`` symbols are now replaced by ``In [x]``, and output, when present, is prepended with ``Out [x]``. If we now type the same commands as before, we get:
# 
#     In [1]: a = 1
# 
#     In [2]: a
#     Out[2]: 1
# 
# If you now type the up arrow twice, you will get back to ``a = 1``.

# ## Running scripts (demo)

# While the interactive Python mode is very useful to exploring and trying out code, you will eventually want to write a script to record and reproduce what you did, or to do things that are too complex to type in interactively (defining functions, classes, etc.). To write a Python script, just use your favorite code editor to put the code in a file with a ``.py`` extension. For example, we can create a file called ``test.py`` containing:
# 
#     a = 1
#     print(a)
# 
# On Linux computers you can use for example the ``emacs`` editor which you can open by typing:
#     
#     emacs &
#     
# (ignore the warnings that it prints to the terminal).
# 
# We can then run the script on the command-line with:
# 
#     $ python test.py
#     1
# 
# Note: The ``print`` statement is necessary, because typing ``a`` on its own will only print out the value in interactive mode. In scripts, the printing has to be explicitly requested with the print command. To print multiple variables, just separate them with a comma after the print command:
# 
#     print(a, 1.5, "spam")

# # Using the IPython notebook

# The IPython *notebook* is allows you to write notebooks similar to e.g. Mathematica. The advantage of doing this is that you can include text, code, and plots in the same document. This makes it ideal for example to write up a report about a project that uses mostly Python code, in order to share with others. In fact, the notes for this course are written using the IPython notebook!

# Click on ``New Notebook`` on the right, which will start a new document. You can change the name of the document by clicking on the **Untitled** name at the top and entering a new name. Make sure you then save the document (make sure that you save regularly as you might lose content if you close the browser window!).
# 
# At first glance, a notebook looks like a fairly typical application - it has a menubar (File, Edit, View, etc.) and a tool bar with icons. Below this, you will see an empty cell, in which you can type any Python code. You can write several lines of code, and once it is ready to run, you can press shift-enter and it will get executed:

# In[ ]:


a = 1
print(a)


# You can then click on that cell, change the Python code, and press shift-enter again to re-execute the code. Once you have executed a cell once, a new cell will appear below. You can again enter some code, then press shift-enter to execute it.

# ### Text

# It is likely that you will want to enter actual text (non-code) in the notebook. To do this, click on a cell, and in the drop-down menu in the toolbar, select 'Markdown'. This is a specific type of syntax for writing text. You can just write text normally and press shift-enter to *render* it:
# 
#     This is some plain text
# 
# To edit it, double click on the cell. You can also enter section headings using the following syntax:
# 
#     This is a title
#     ===============
# 
#     This is a sub-title
#     -------------------
# 
# which will look like:
# 
# This is a title
# ===============
# 
# This is a sub-title
# -------------------
# 
# Finally, if you are familiar with LaTeX, you can enter equations using:
# 
#     $$E = m c^2$$
# 
# on a separate line, or:
# 
#     The equation $p=h/\lambda$ is very important
# 
# to include it in a sentence. This will look like:
# 
# $$E = m c^2$$
# 
# The equation $p=h/\lambda$ is very important
# 
# For more information about using LaTeX for equations, see [this guide](http://en.wikibooks.org/wiki/LaTeX/Mathematics).

# ## A very first Python program
# 
# A Python program can be a single line:

# In[2]:


print("Hello interweb")


# ## Loops in Python

# Why not make a `for` loop with some formatted output:

# In[5]:


for n in range(3):
    print(f"Hello interweb, this is iteration number {n}")


# A few things to note:
# 
# - Python defaults to counting from 0 (like C) rather than from 1 (like Fortran).
# - Function calls in Python always use parentheses: `print()`
# - The colon `:` denotes the beginning of a definition (here of the repeated code under the `for` loop.
# - Code blocks are identified through indentations.
# 
# To emphasize this last point, here an example with a two-line repeated block:

# In[7]:


for n in range(3):
    print("Hello interweb!")
print(f"This is iteration number {n}.")
print('And now we are done.')


# In[11]:


n = [2, 3, 5, 7]

for e in n:
    print(e)


# > Please avoid Matlab-like for statements with range

# In[10]:


for e in range(len(n)):
    print(n[e]) 


# In[71]:


m = range(5)  # m = [0, 1, 2, 3, 4]
for _ in m:
    print("hola")


# In[13]:



for index, value in enumerate(n):
# for value in n:
# for index in range(len(n)):
    print("The value of index ", index, " is ", value)


# In[19]:


kk = 0
for value in n:
    print(value)
    print(kk)
    #print(f'{value} - {kk}')
    kk = kk + 1 # kk += 1
    
for kk, value in enumerate(n):
    print(f'{value} - {kk}')
    


# ## Basic flow control
# 
# Like most languages, Python has an `if` statement for logical decisions:

# In[24]:


n = 1
if n > 2:
    print("n is greater than 2!")
#else:
#    print("n is not greater than 2!")


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


# Python also defines the `True` and `False` logical constants:

# In[25]:


n > 2


# In[26]:


result = 4 < 5
print(result)


# There's also a `while` statement for conditional looping:

# In[27]:


m = 0
while m < 3:
    print(f"This is iteration number {m}.")
    m += 1
print(m < 3)


# ## Basic Python data types
# 
# Python is a very flexible language, and many advanced data types are introduced through packages (more on this below). But some of the basic types include: 

# ## Built-In Types

# | Type        | Example        | Description                                                  |
# |-------------|----------------|--------------------------------------------------------------|
# | ``int``     | ``x = 1``      | integers (i.e., whole numbers)                               |
# | ``float``   | ``x = 1.0``    | floating-point numbers (i.e., real numbers)                  |
# | ``complex`` | ``x = 1 + 2j`` | Complex numbers (i.e., numbers with real and imaginary part) |
# | ``bool``    | ``x = True``   | Boolean: True/False values                                   |
# | ``str``     | ``x = 'abc'``  | String: characters or text                                   |
# | ``NoneType``| ``x = None``   | Special object indicating nulls                              |

# ### Integers (`int`)
# 
# The number `m` above is a good example. We can use the built-in function `type()` to inspect what we've got in memory:

# In[28]:


m = 4


# In[29]:


type(m)


# ### Floating point numbers (`float`)
# 
# Floats can be entered in decimal notation:

# In[30]:


type(0.1)


# In[34]:


n = 5 / 2


# In[35]:


n


# In[33]:


type(n)


# or in scientific notation:

# In[36]:


type(4e7)


# where `4e7` is the Pythonic representation of the number $ 4 \times 10^7 $.

# ### Character strings (`str`)
# 
# You can use either single quotes `''` or double quotes `" "` to denote a string:

# In[37]:


type("orange")


# In[38]:


type('orange')


# <center> <img src="img/list-indexing.png" width="1600"/> </center>

# In[39]:


p = 'orange'


# In[40]:


p[-1]


# In[42]:


message = "what do_you-like?"
print(message)


# You can use either single quotes (``'``), double quotes (``"``), or triple quotes (``'''`` or ``"""``) to enclose a string (the last one is used for multi-line strings). To include single or double quotes inside a string, you can either use the opposite quote to enclose the string:

# In[43]:


response = "I'm"
print(response)


# In[44]:


response = 'hello'
print(response)


# In[45]:


# length of string
print(len(response))


# In[47]:


# Make upper-case. See also str.lower()
print(response.lower())


# In[41]:


# Capitalize. See also str.title()
print(message.capitalize())


# In[56]:


s = "Spam egg spam spam"
s.index('e')  # An integer giving the position of the sub-string


# In[57]:


s2 = s.split()
s2


# In[58]:


s2[2].split('p')


# In[60]:


s2 = "Spam-egg-spam_spam"
s2.split('_')


# In[61]:


s3 = s2.split('-')
s3


# In[63]:


type(s3[1])


# In[67]:


"_".join("-".join(['hola', 'adios']).split('o'))


# In[70]:


vv = 5.0000045


# In[71]:


# concatenation with +
print(message + ' ' + response + ' ' +str(vv))


# In[75]:


type(vv)


# ### FORMAT
# 
# # a = f'{message} {response} {vv}'
# 
# # a = '{} {} {}'.format(message, response, vv)
# 
# # a = '{0} {2} {1}'.format(message, response, vv)

# In[77]:



# concatenation with f
print(f'{message} {response} {vv:.2f}')


# In[78]:


f'{5:03d}'


# In[78]:


# concatenation with format
print('{} {} {}'.format(message, response, vv))
print('{1} -- {0} {1}'.format(message, response))


# In[79]:


# multiplication is multiple concatenation
print(5 * response)


# In[81]:


message


# In[86]:


message[0:8:2]


# In[103]:


# Access individual characters (zero-based indexing)
#[first:end:step]
print(message[:8:2])


# In[87]:


cadena1 = "cool"
print(cadena1[::-1])


# ## Exercise 
# Given a string such as the one below, make a new string that does not contain the word ``egg``:

# In[91]:


a = "Hello, egg world!"

# enter your solution here
# "Hello world!"


# In[93]:


b = a.split(' e')
c = a.split('g ')
print(b[0] + ' ' + c[1])
print(f'{b[0]} {c[1]}')


# In[96]:


b = a.index(' e')
c = a.index(' w')
print(a[:b] + a[c:])


# In[99]:


''.join(a.split(' egg'))


# In[100]:


print(' '.join(a.split(', egg ')))


# In[102]:


s = a.split(' ')
print(s[0] + ' ' + s[2])


# ## Built-In Data Structures

# | Type Name | Example                   |Description                            |
# |-----------|---------------------------|---------------------------------------|
# | ``list``  | ``[1, 2, 3]``             | Ordered collection                    |
# | ``tuple`` | ``(1, 2, 3)``             | Immutable ordered collection          |
# | ``dict``  | ``{'a': 1, 'b': 2, 'c': 3}`` | (key,value) mapping                |

# ### Lists
# 
# A list is an ordered container of objects denoted by **square brackets**:

# In[103]:


mylist = [0, 1, 1, 2, 3, 5, 8]


# In[105]:


len(mylist)


# In[106]:


mylist


# In[111]:


mylist[0:6:2]


# In[113]:


ppp = list()
for n in range(5):
    ppp.append(n+2)
ppp


# In[114]:


for p in ppp:
    print(p + 1)


# Lists are useful for lots of reasons including iteration:

# In[116]:


for pos, number in enumerate(mylist):
    print(f'{pos + 1} - {number + 4.5}')
    
# for pos in range(len(mylist)):
#    print(mylist[pos])


# Lists do **not** have to contain all identical types:

# In[117]:


myweirdlist = [0, 1, 1, "apple", 4e7]
for item in myweirdlist:
    print(f'{item} - {type(item)}')


# In[119]:


ll = [['a', 'be ready', 'c'], [23, 45]]


# In[127]:


ll[1][1] +1


# In[148]:


ll[0][1].split()[1][3]


# In[155]:


ll[0][1].split()[1][2:3]


# This list contains a mix of `int`, `float`, and `str` (character string).

# Because a list is *ordered*, we can access items by integer index:

# In[ ]:


myweirdlist[3]


# remembering that we start counting from zero!

# Python also allows lists to be created dynamically through *list comprehension* like this:

# In[132]:


squares = [i ** 2 for i in range(11)]
squares


# In[131]:


squares = list()
for i in range(11):
    squares.append(i**2)
squares


# In[133]:


li = [2, 3, 5, 7]


# In[134]:


# Change value of list
li[1] = -2.2


# In[135]:


li


# In[137]:


# Append a value to the end
li.append(11)
print(li)


# In[139]:


lst = li


# In[140]:


# Addition concatenates lists
print(lst + [13, 17, 19])


# In[141]:


lst.remove(5)
lst


# In[149]:


lst.


# In[150]:


lst


# In[152]:


# sort() method sorts in-place
lst = [2, 5, 1, 6, 3, 4]
lst.sort()
print(lst)


# In[173]:


lst = [1, "two", 3.14, [0, 3, 5]]
print(lst)


# In[178]:


type(lst[3])


# #### List indexing and slicing

# In[179]:


lst = [2, 3, 5, 7, 11]


# In[180]:


print(lst[0])


# In[181]:


print(lst[1])


# In[182]:


print(lst[-1])


# In[183]:


print(lst[0:3])


# In[184]:


print(lst[:3])


# In[186]:


print(lst[::-2])  # [start:end:step] equivalent to l[0:len(l):2]


# In[58]:


lst[0] = 100
print(lst)


# ### Dictionaries (`dict`)
# 
# A dictionary is a collection of *labeled objects*. Python uses curly braces `{}` to create dictionaries:

# In[163]:


mypet = {
    "name": {'male': ["Fluffy", 'Fido'], 'female':'Fida'},
    "species": ["cat", 'dog', 'parrot', 'iguana'],
    "age": [4, 5],
}
type(mypet)


# In[164]:


mypet


# We can then access items in the dictionary by label using square brackets:

# In[166]:


mypet["name"]['female']


# We can iterate through the keys (or labels) of a `dict`:

# In[158]:


for key, values in mypet.items():
    print("The key is:", key)
    print("The value is:", values)


# In[167]:


# Set a new key:value pair
mypet["ninety"] = 90
print(mypet)


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

# In[169]:


info = {'Albert Einstein': '03/14/1879',
        'Benjamin Franklin': '01/17/1706',
        'Ada Lovelace': '12/10/1815',
        'Marie Curie': '07/11/1867',
        'Rowan Atkinson': '01/6/1955',
        'Rosalind Franklin': '25/07/1920'}

info2 = {'name': ['Albert Einstein', 'Benjamin Franklin', 'Ada Lovelace', 'Marie Curie', 'Rowan Atkinson', 
                  'Rosalind Franklin'],
         'birthday': ['03/14/1879', '01/17/1706', '12/10/1815', '07/11/1867', '01/6/1955','25/07/1920']}


# In[187]:


if 'Andrea' in info:
    print('Marie Curie is present')


# In[179]:


a = info['Albert Einstein']
a


# In[188]:


if info['Marie Curie']:
    print("She's there")
   
    


# In[185]:


info['Marie Curie']


# In[222]:


# Check if Marie Curie is in dictionary
a = 'Albert Einstein'

if a in info:    
    #ksjklajkld
    print(info)


# In[ ]:





# In[210]:


print(a in info)


# In[212]:


info2['name'].index('Marie Curie')


# In[217]:


# Einstein's birthday
'03/14/1879'


# In[218]:


info['Albert Einstein']


# In[220]:


pos_e = info2['name'].index('Albert Einstein')


# In[221]:


info2['birthday'][pos_e]


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

# In[15]:


print(22 / 2 == 10 + 1)


# In[191]:


# 25 is even
print(25 % 2 != 0)


# In[190]:


# 66 is odd
print(66 % 2 == 0)


# In[18]:


# check if a is between 15 and 30
a = 25
print(15 < a < 30)


# ## Arrays of numbers with `numpy`
# 
# The vast majority of scientific Python code makes use of *packages* that extend the base language in many useful ways.
# 
# Almost all scientific computing requires ordered arrays of numbers, and fast methods for manipulating them. That's what numpy does in the Python world.
# 
# Using any package requires an `import` statement, and (optionally) a nickname to be used locally, denoted by the keyword `as`:

# In[4]:


import numpy as np


# Now all our calls to `numpy` functions will be preceeded by `np.`

# Create a linearly space array of numbers:

# In[5]:


# linspace() takes 3 arguments: start, end, total number of points
numbers = np.linspace(0.0, 1.0, 11)
numbers


# We've just created a new type of object defined by numpy:

# In[6]:


type(numbers)


# Do some arithmetic on that array:

# In[8]:


numbers + 1


# In[2]:


list = [3, 4, 5]
import numpy as np


# In[6]:


list + list


# Sum up all the numbers:

# In[ ]:


np.sum(numbers)


# ### Boolean Operations

# In[24]:


x = 4
print(((x < 6) and (x > 2)) and (x == 5))
# and -> &


# In[25]:


print((x > 10) or (x % 2 == 0))
# or -> |


# In[19]:


print(not (x < 6))


# ### Membership Operators

# | Operator      | Description                                       |
# |---------------|---------------------------------------------------|
# | ``a in b``    | True if ``a`` is a member of ``b``                |
# | ``a not in b``| True if ``a`` is not a member of ``b``            |

# In[26]:


print(1 in [1, 2, 3])


# In[27]:


print(2 not in [1, 2, 3])


# ## Functions

# ### Defining Functions

# In[33]:


import time

# def NAME(input1, input 2):
#     aaaaa
#     bbbbb

#     return output1, output2   

# out = NAME(inp1, inp2)
# out = [out1, out2]

def header():
    text = "This is a function"
    text += ". Copyright " + time.strftime("%d-%m-%Y")

    return text, text 


out1, _ = header()

#print(header())
print(out1)


# In[ ]:





# In[35]:


def header_v2(author):
    text = "This function is written by "
    text += author
    text += f". Copyright {time.strftime('%d-%m-%Y')}"

    return text


print(header_v2("Andrea Lira Loarca"))


# In[39]:


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


# In[40]:


print(fibonacci(10))


# ### Default Argument Values

# In[45]:


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


# In[46]:


print(fibonacci(10))


# In[47]:


print(fibonacci(10, 5))


# In[83]:


## Keyword arguments


# In[51]:


print(fibonacci(start=5, n=20))


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

# In[57]:


import numpy as np
from numpy import linspace


# In[58]:


linspace(0, 5, 10)


# #### Explicit import of module contents

# In[59]:


import scipy.stats as st
from scipy.stats import norm, genpareto

