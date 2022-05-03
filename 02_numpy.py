#!/usr/bin/env python
# coding: utf-8

# # Introduction to Numpy

# Python lists:
# 
# * are very flexible
# * don't require uniform numerical types
# * are very easy to modify (inserting or appending objects).
# 
# However, flexibility often comes at the cost of performance, and lists are not the ideal object for numerical calculations.

# This is where **Numpy** comes in. Numpy is a Python module that defines a powerful n-dimensional array object that uses C and Fortran code behind the scenes to provide high performance.

# In[2]:


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


# In[3]:


import numpy

start_time = time.time()
a = numpy.array(a)
new_a = func(a)
print(f'{time.time()-start_time} seconds')


# The downside of Numpy arrays is that they have a more rigid structure, and require a single numerical type (e.g. floating point values), but for a lot of scientific work, this is exactly what is needed.

# The Numpy module is imported with:

# In[1]:


import numpy


# Although in the rest of this course, and in many packages, the following convention is used:

# In[2]:


import numpy as np


# This is because Numpy is so often used that it is shorter to type ``np`` than ``numpy``.

# ## Creating Numpy arrays

# The easiest way to create an array is from a Python list, using the ``array`` function:

# In[13]:


a = np.array([10.1, 20.3, 30, 40])


# In[6]:


type(a)


# In[8]:


L = range(1000)
get_ipython().run_line_magic('timeit', '[i**2 for i in L]')

a = np.arange(1000)
get_ipython().run_line_magic('timeit', 'a**2')


# In[9]:


L


# Numpy arrays have several attributes that give useful information about the array:

# In[10]:


a.ndim  # number of dimensions


# In[11]:


a.shape  # shape of the array


# In[14]:


a.dtype  # numerical type


# There are several other ways to create arrays. For example, there is an ``arange`` function that can be used similarly to the built-in Python ``range`` function, with the exception that it can take floating-point input:

# In[15]:


np.arange(10.5)
# np.arange(0, 10, 1)


# In[22]:


np.arange(30.5)


# In[13]:


np.arange(3, 12, 2)


# In[31]:


b = np.arange(1.2, 4.41, 0.1)
b


# In[30]:


b[3]


# In[ ]:





# Another useful function is ``linspace``, which can be used to create linearly spaced values between and including limits:

# In[27]:


np.linspace(11, 12, 10)


# and a similar function can be used to create logarithmically spaced values between and including limits:

# In[32]:


np.logspace(1., 4., 7)


# Finally, the ``zeros`` and ``ones`` functions can be used to create arrays intially set to ``0`` and ``1`` respectively:

# In[33]:


np.zeros(10)


# In[34]:


np.ones(5)


# ## Exercise

# Create an array which contains 11 values logarithmically spaced between $10^{-20}$ and $10^{-10}$.

# In[47]:


# your solution here
np.logspace(-20,-10,11)


# Create an array which contains the value 2 repeated 10 times

# In[52]:


# your solution here
b=np.linspace(2,2,10)
b


# In[54]:


np.zeros(10)+2


# Try using ``np.empty(10)`` and compare the results to ``np.zeros(10)`` - why do you think there is a difference?

# In[48]:


np.empty(20)


# In[49]:


# your solution here
a = np.empty(20)
print(a)


# In[50]:


b = np.zeros(20)
print(b)


# In[51]:


a==b


# Create an array containing 5 times the value 0, as a 32-bit floating point array (this is harder)

# In[22]:


# your solution here


# ## Combining arrays

# Numpy arrays can be combined numerically using the standard ``+-*/**`` operators:

# In[55]:


x1 = np.array([1,2,3])
y1 = np.array([4,5,6])


# In[56]:


2 * y1


# In[59]:


(x1 + 2) * y1


# In[60]:


x1 ** y1


# Note that this differs from lists:

# In[61]:


x = [1,2,3]
y = [4,5,6]


# In[62]:


y.append('hola')
y


# In[63]:


3 * y


# In[65]:


x + 2 * y


# ## Accessing and Slicing Arrays

# Similarly to lists, items in arrays can be accessed individually:

# In[67]:


x = np.array([9,8,7])


# In[68]:


x


# In[69]:


x[0]


# In[70]:


x[1]


# and arrays can also be **sliced** by specifiying the start and end of the slice (where the last element is exclusive):

# In[72]:


y = np.arange(10, 20)
y


# In[73]:


y[0:5] # [start:end:step]


# optionally specifying a step:

# In[74]:


y[0:10:2]


# As for lists, the start, end, and step are all optional, and default to ``0``, ``len(array)``, and ``1`` respectively:

# In[77]:


y[:5]


# In[78]:


y[::2]


# ## Exercise

# Given an array ``x`` with 20 elements, find the array ``dx`` containing 19 values where ``dx[i] = x[i+1] - x[i]``. Do this without loops!

# In[83]:


[4, 6, 7]
# i + 1 = [6, 7]
# i = [4, 6]
# [2, 1]

a = np.linspace(6, 17, 25)
a


# In[84]:


a[:19]


# In[90]:


dx = a[1:]-a[:-1]
dx


# ## Multi-dimensional arrays

# <center> <img src="img/numpy_indexing.png" width="1600"/> </center>

# In[ ]:


a[3:5, :6:2]


# Numpy can be used for multi-dimensional arrays:

# In[97]:


x = np.array([[1.,2.],[3.,4.]])


# In[ ]:


x.


# In[98]:


x.ndim


# In[99]:


x.shape


# In[100]:


y = np.ones([3,2,3])  # ones takes the shape of the array, not the values


# In[101]:


y


# In[102]:


y.shape


# Multi-dimensional arrays can be sliced differently along different dimensions:

# In[43]:


z = np.ones([6,6,6])


# In[44]:


z[::3, 1:4, :]


# ## Constants

# NumPy provides us access to some useful constants as well - remember you should never be typing these in manually! Other libraries such as SciPy and MetPy have their own set of constants that are more domain specific.

# In[ ]:


np.pi


# In[ ]:


np.e


# ## Functions

# In addition to an array class, Numpy contains a number of **vectorized** functions, which means functions that can act on all the elements of an array, typically much faster than could be achieved by looping over the array.

# For example:

# In[45]:


theta = np.linspace(0., 2. * np.pi, 10)


# In[46]:


theta


# In[47]:


np.sin(theta)


# Another useful package is the ``np.random`` sub-package, which can be used to genenerate random numbers fast:

# In[48]:


# uniform distribution between 0 and 1
np.random.random(10)


# In[1]:


# 10 values from a gaussian distribution with mean 3 and sigma 1
np.random.normal(3., 1., 10)


# In[2]:


a = np.arange(12).reshape(3, 4)
a


# In[3]:


np.sum(a)


# In[ ]:


a.sum()


# In[ ]:


a.shape


# In[ ]:


np.sum(a, axis=1)


# In[ ]:





# In[ ]:





# Another very useful function in Numpy is [numpy.loadtxt](http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html) which makes it easy to read in data from column-based data. For example, given the following file:

# In[104]:


data = np.loadtxt('data/columns.txt')
data.ndim


# Or we can read the individual columns:

# In[106]:


_, temperature = np.loadtxt('data/columns.txt', unpack=True)


# In[107]:





# In[53]:


temperature


# There are additional options to skip header rows, ignore comments, and read only certain columns. See the documentation for more details.

# In[ ]:





# ## Masking

# The index notation ``[...]`` is not limited to single element indexing, or multiple element slicing, but one can also pass a discrete list/array of indices:

# In[109]:


x = np.array([1,6,4,7,9,3,1,5,6,7,3,4,4,3])
x[[1,2,4,3,3,2]]


# which is returning a new array composed of elements 1, 2, 4, etc from the original array.

# Alternatively, one can also pass a boolean array of ``True/False`` values, called a **mask**, indicating which items to keep:

# In[110]:


x[np.array([True, False, False, True, True, True, False, False, True, True, True, False, False, True])]


# Now this doesn't look very useful because it is very verbose, but now consider that carrying out a comparison with the array will return such a boolean array:

# In[111]:


x > 3.4


# It is therefore possible to extract subsets from an array using the following simple notation:

# In[112]:


x[x > 3.4]


# Conditions can be combined:

# In[113]:


x[(x > 3.4) & (x < 5.5)]


# Of course, the boolean **mask** can be derived from a different array to ``x`` as long as it is the right size:

# In[125]:


x = np.linspace(-1., 1., 14)
y = np.array([1,6,4,7,9,3,1,5,6,7,3,4,4,3])


# In[117]:


y2 = y + 3
y2


# In[119]:


yy = y[y2 >= 9]
yy


# In[121]:


y[(x > -0.5) | (x < 0.4)]


# Since the mask itself is an array, it can be stored in a variable and used as a mask for different arrays:

# In[61]:


keep = (x > -0.5) & (x < 0.4)
x_new = x[keep]
y_new = y[keep]


# In[62]:


x_new


# In[63]:


y_new


# we can use this conditional indexing to assign new values to certain positions within our array, somewhat like a masking operation.

# In[122]:


y


# In[132]:


y[y > 5] = 3


# In[4]:


y


# In[ ]:


y[y > 5] = 3


# In[ ]:





# In[ ]:





# ### NaN values

# In arrays, some of the values are sometimes NaN - meaning *Not a Number*. If you multiply a NaN value by another value, you get NaN, and if there are any NaN values in a summation, the total result will be NaN. One way to get around this is to use ``np.nansum`` instead of ``np.sum`` in order to find the sum:

# In[134]:


x = np.array([1,2,3,np.nan])


# In[135]:


np.sum(x)


# In[67]:


np.nansum(x)


# In[68]:


np.nanmax(x)


# You can also use ``np.isnan`` to tell you where values are NaN. For example, ``array[~np.isnan(array)]`` will return all the values that are not NaN (because ~ means 'not'):

# In[69]:


np.isnan(x)


# In[70]:


x[np.isnan(x)]


# In[71]:


x[~np.isnan(x)]


# ### Statistics --> Scipy

# In[72]:


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


# ## Exercise

# The [data/SIMAR_gaps.txt](data/SIMAR_gaps.txt) data file gives the wave climate data in the Mediterranean Sea.

# Read in the file using ``np.loadtxt``. The data contains bad values, which you can identify by looking at the minimum and maximum values of the array. Use masking to get rid of the bad temperature values.

# In[148]:


data = np.loadtxt('data/SIMAR_gaps.txt', skiprows=1)
data


# In[150]:


H = data[:, 4]
print(np.min(H))
print(np.max(H))


# In[149]:


data_fil = data[H>-99.9, :]
H_fil = H[H>-99.9]


# In[143]:


data.shape


# In[146]:


data_fil.shape
H_fil = data_fil[:, 4]
np.min(H_fil)


# In[74]:



# your solution here
data = np.loadtxt('data/SIMAR_gaps.txt', skiprows=1)
print(data)
Hm0 = data[:, 4]
print(Hm0.min())

data[Hm0>-99.9, :]


# ### Linear Algebra

# In[73]:


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

