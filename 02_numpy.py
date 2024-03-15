# %% [markdown]
# # Introduction to Numpy

# %% [markdown]
# Python lists:
# 
# * are very flexible
# * don't require uniform numerical types
# * are very easy to modify (inserting or appending objects).
# 
# However, flexibility often comes at the cost of performance, and lists are not the ideal object for numerical calculations.

# %% [markdown]
# This is where **Numpy** comes in. Numpy is a Python module that defines a powerful n-dimensional array object that uses C and Fortran code behind the scenes to provide high performance.

# %%
import time
import math

a = range(10000000)

def func(a):
    return 1e-6*(4*a**2.) #7+1./(a+3)**3.4-(23*a)**4.2)-20

# measure how long it takes in seconds
start_time = time.time()

new_a = []
for val in a:
    new_a.append(func(val))
    
print(f'{time.time()-start_time} seconds')

# %%
a

# %%
import numpy

start_time = time.time()
a = numpy.array(a)
new_a = func(a)
print(f'{time.time()-start_time} seconds')

# %%
L = range(1000)
%timeit [i**2 for i in L]

a = np.arange(1000)
%timeit a**2

# %% [markdown]
# The downside of Numpy arrays is that they have a more rigid structure, and require a single numerical type (e.g. floating point values), but for a lot of scientific work, this is exactly what is needed.

# %% [markdown]
# The Numpy module is imported with:

# %%
import numpy

# %% [markdown]
# Although in the rest of this course, and in many packages, the following convention is used:

# %%
import numpy as np

# %% [markdown]
# This is because Numpy is so often used that it is shorter to type ``np`` than ``numpy``.

# %% [markdown]
# ## Creating Numpy arrays

# %% [markdown]
# The easiest way to create an array is from a Python list, using the ``array`` function:

# %%
a = np.array([10, 20.1, 30, 40])

# %%
b = list()
for n in range(5):
    b.append(n)
c = np.array(b)
c

# %%
type(a)

# %% [markdown]
# Numpy arrays have several attributes that give useful information about the array:

# %%
a.ndim  # number of dimensions

# %%
a.shape  # shape of the array

# %%
a.dtype  # numerical type

# %% [markdown]
# There are several other ways to create arrays. For example, there is an ``arange`` function that can be used similarly to the built-in Python ``range`` function, with the exception that it can take floating-point input:

# %%
d =np.arange(1, 6, 1)
d
# np.arange(0, 10, 1)

# %%
d.dtype

# %%
np.arange(30.5)

# %%
np.arange(3, 12, 2)

# %%
b = np.arange(1.2, 4.4, 0.1)


# %%
b

# %%
b[3]

# %% [markdown]
# Another useful function is ``linspace``, which can be used to create linearly spaced values between and including limits:

# %%
np.linspace(11, 12, 10)

# %% [markdown]
# and a similar function can be used to create logarithmically spaced values between and including limits:

# %%
np.logspace(1., 4., 7)

# %% [markdown]
# Finally, the ``zeros`` and ``ones`` functions can be used to create arrays intially set to ``0`` and ``1`` respectively:

# %%
np.zeros(10)

# %%
np.ones([5, 3, 5])[0, :, 1:3]

# %% [markdown]
# np.array([X, X, X, X])
# 
# np.arange(start, finish, step)
# 
# np.linspace(start, finish-included, number of elements)
# 
# np.zeros([dim, dim]) 1D: np.zeros(X)
# 
# np.ones([dim, dim])
# 
# np.empty([dim, dim])

# %%
a 

# %%
b

# %%
c = np.vstack([a, a]) ### stack, vstack, hstack
c.shape

# %%


# %%
np.shape(c)

# %%
d = np.vstack([a, a])
d

# %%
e = np.hstack([a, b])

# %%
np.shape(e)

# %% [markdown]
# ## Exercise

# %% [markdown]
# Create an array which contains the value 2 repeated 10 times

# %%


# %%


# %% [markdown]
# Create an array which contains values from 1 until 90 every one and then the values 95, 99, 99.9, 99.99.

# %%


# %%


# %%


# %% [markdown]
# ## Numerical operations with arrays

# %% [markdown]
# Numpy arrays can be combined numerically using the standard ``+-*/**`` operators:

# %%
x1 = np.array([1,2,3])
y1 = np.array([4,5,6])

# %%
y1

# %%
2 * y1

# %%
(x1 + 2) * y1

# %%
x1 ** y1

# %% [markdown]
# Note that this differs from lists:

# %%
x = [1,2,3]
y = [4,5,6]

# %%
y.append('hola')
y

# %%
3 * y

# %%
2 * y

# %%
x + 2 * y

# %% [markdown]
# ## Accessing and Slicing Arrays

# %% [markdown]
# Similarly to lists, items in arrays can be accessed individually:

# %%
x = np.array([9,8,7])

# %%
x

# %%
x[0]

# %%
x[1]

# %% [markdown]
# and arrays can also be **sliced** by specifiying the start and end of the slice (where the last element is exclusive):

# %%
y = np.arange(10, 20)
y

# %%
y[0:5:2] # Slices [start:end:step]

# %% [markdown]
# optionally specifying a step:

# %%
y[0:10:2]

# %% [markdown]
# As for lists, the start, end, and step are all optional, and default to ``0``, ``len(array)``, and ``1`` respectively:

# %%
y[:5]

# %%
y[::2]

# %% [markdown]
# ## Exercise

# %% [markdown]
# Given an array ``x`` with 20 elements, find the array ``dx`` containing 19 values where ``dx[i] = x[i+1] - x[i]``. Do this without loops!

# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Multi-dimensional arrays

# %% [markdown]
# <center> <img src="img/numpy_indexing.png" width="1600"/> </center>

# %%
y = np.ones([3,2,3])  # ones takes the shape of the array, not the values

# %%
y.shape

# %%
y

# %% [markdown]
# Multi-dimensional arrays can be sliced differently along different dimensions:

# %%
z = np.ones([6,6,6])

# %%
z[::3, 1:4, :]

# %% [markdown]
# ## Constants

# %% [markdown]
# NumPy provides us access to some useful constants as well - remember you should never be typing these in manually! Other libraries such as SciPy and MetPy have their own set of constants that are more domain specific.

# %%
np.pi

# %%
np.e

# %%
1 + np.pi

# %% [markdown]
# ## Functions

# %% [markdown]
# In addition to an array class, Numpy contains a number of **vectorized** functions, which means functions that can act on all the elements of an array, typically much faster than could be achieved by looping over the array.

# %% [markdown]
# For example:

# %%
theta = np.linspace(0., 2. * np.pi, 10)

# %%
theta.shape

# %%
np.shape(theta)

# %%
theta.max()

# %%
np.max(theta)

# %%
aa = np.array([np.linspace(0., 2. * np.pi, 10), np.linspace(0., 2. * np.pi, 10)])

# %%
pp = [np.linspace(0., 2. * np.pi, 10), np.linspace(0., 2. * np.pi, 10)]
pp

# %%
type(pp)

# %%
aa = np.asarray(pp)

# %%
aa.shape

# %%
np.sin(aa)

# %%
np.sin(aa[0, 0])

# %% [markdown]
# Another useful package is the ``np.random`` sub-package, which can be used to genenerate random numbers fast:

# %%
# uniform distribution between 0 and 1
np.random.random(10)

# %%
# 10 values from a gaussian distribution with mean 3 and sigma 1
np.random.normal(3., 1., 10)

# %%
a = np.arange(12).reshape(3, 4)
a

# %%
a

# %%
np.sum(a, axis=0)

# %%
a.shape

# %%
np.sum(a, axis=1)

# %% [markdown]
# Another very useful function in Numpy is [numpy.loadtxt](http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html) which makes it easy to read in data from column-based data. For example, given the following file:

# %%
from pathlib import Path

dir_data = Path('data')
data = np.loadtxt(dir_data / 'columns.txt')

# %%
data

# %% [markdown]
# ## Masking

# %% [markdown]
# The index notation ``[...]`` is not limited to single element indexing, or multiple element slicing, but one can also pass a discrete list/array of indices:

# %%
x = np.array([1,6,4,7, 9, 8, 10])
#x[[True, False, True, False]]

# %% [markdown]
# which is returning a new array composed of elements 1, 2, 4, etc from the original array.

# %% [markdown]
# Alternatively, one can also pass a boolean array of ``True/False`` values, called a **mask**, indicating which items to keep:

# %%
y = np.array([3, 4, 5])

# %%
mask = np.array([True, False, False])

# %%
y[mask]

# %% [markdown]
# Now this doesn't look very useful because it is very verbose, but now consider that carrying out a comparison with the array will return such a boolean array:

# %%
x

# %%
mask = x > 3.4

# %% [markdown]
# It is therefore possible to extract subsets from an array using the following simple notation:

# %%
x[x > 3.4]

# %%
x[mask]

# %%
x[~mask]

# %% [markdown]
# Conditions can be combined:

# %% [markdown]
# ### Conditional formating
# 
# ##### Loops
# and, or
# 
# #### Masking in numpy array
# & (and), | (or)

# %%
x

# %%
x[(x > 3.4) & (x < 5.5)]

# %% [markdown]
# Of course, the boolean **mask** can be derived from a different array to ``x`` as long as it is the right size:

# %%
x = np.linspace(-1., 1., 14)
y = np.array([1,6,4,7,9,3,1,5,6,7,3,4,4,3])

# %%
y.shape

# %%
x.shape

# %%
y2 = y + 3
y2

# %%
mask = y2 >= 9
yy = y[mask]
yy

# %%
y[(x > -0.5) | (x < 0.4)]

# %% [markdown]
# Since the mask itself is an array, it can be stored in a variable and used as a mask for different arrays:

# %%
keep = (x > -0.5) & (x < 0.4)
x_new = x[keep]
y_new = y[keep]

# %%
keep

# %%
x_new

# %%
y_new

# %% [markdown]
# we can use this conditional indexing to assign new values to certain positions within our array, somewhat like a masking operation.

# %%
y

# %%
mask = y>5
mask

# %%
y[mask] = 999

# %%
y

# %%
y1 = y[mask]
y1

# %%
mm = y > 3

# %%
y[mm] = np.nan

# %%
y[y > 5] = 3

# %% [markdown]
# ### NaN values

# %% [markdown]
# In arrays, some of the values are sometimes NaN - meaning *Not a Number*. If you multiply a NaN value by another value, you get NaN, and if there are any NaN values in a summation, the total result will be NaN. One way to get around this is to use ``np.nansum`` instead of ``np.sum`` in order to find the sum:

# %%
x = np.array([1,2,3,np.NaN])
x

# %%
np.NAN # np.nan | np.NaN | np.NAN

# %%
np.nansum(x)

# %%
np.nansum(x)

# %%
np.nanmax(x)

# %% [markdown]
# You can also use ``np.isnan`` to tell you where values are NaN. For example, ``array[~np.isnan(array)]`` will return all the values that are not NaN (because ~ means 'not'):

# %%
np.isnan(x)

# %%
x[np.isnan(x)]

# %%
x[~np.isnan(x)]

# %% [markdown]
# ### Statistics --> Scipy

# %%
import numpy.random as rnd


g = rnd.normal(loc=0, scale=1, size=1000000)

print(numpy.mean(g), numpy.median(g), numpy.std(g))

# specifying axis of operation gives different results:
a = [[1,1,1], [2,2,2], [3,3,3]]
print(numpy.mean(a))         # mean of all numbers in the array
print(numpy.mean(a, axis=0)) # mean along axis 0 (first axis = outermost axis = along columns)
print(numpy.mean(a, axis=1)) # mean along axis 1 (second axis = along rows)

# operations that ignore nans
b = [1, 2, 3, numpy.nan, 4, 5]
print(numpy.mean(b))    # returns nan
print(numpy.nanmean(b)) # ignores nan; see nanmedian, nanstd, ...

# determine percentiles
print(numpy.percentile(g, 50)) # the same as median
print(numpy.percentile(g, 68.27)-numpy.percentile(g, 31.73))

# create a histogram
hist, bins = numpy.histogram(g, bins=numpy.arange(-5, 6, 1))
print(bins)
print(hist)

# %% [markdown]
# ## Exercise

# %% [markdown]
# The [data/SIMAR_gaps.txt](data/SIMAR_gaps.txt) data file gives the wave climate data in the Mediterranean Sea.

# %% [markdown]
# Read in the file using ``np.loadtxt``. The data contains bad values, which you can identify by looking at the minimum and maximum values of the array. Use masking to get rid of the bad values.

# %%
from pathlib import Path
dir_data = Path('data')
data = np.loadtxt(dir_data / 'SIMAR_gaps.txt', skiprows=1)
#np.genfromtxt()
var = data[:, 4]

# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ### Linear Algebra

# %%
import numpy
import numpy.linalg as la

a = [1,2,3] 
b = [4,5,6]

print(numpy.dot(a,b)) # dot product
print(numpy.inner(a,b)) 
print(numpy.outer(a,b)) 

i = numpy.diag([1,2,3])
print(la.eig(i)) # return eigenvalues and eigenvectors
print(la.det(i)) # return determinant

# solve linear equations
a = [[3,2,-1], [2,-2,4], [-1,0.5,-1]]
b = [1,-2,0]
print(la.solve(a,b)) 


