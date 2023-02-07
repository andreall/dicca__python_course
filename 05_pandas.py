#!/usr/bin/env python
# coding: utf-8

# # Pandas

# At the very basic level, Pandas objects can be thought of as enhanced versions of NumPy structured arrays in which the rows and columns are identified with labels rather than simple integer indices.

# ### Numpy:
# Data WITHOUT indexes ---> Arrays (1D or many dimensions)

# <center> <img src="img/numpy_arrays.png" width="600"/> </center>

# ### Pandas:
# Data WITH indexes ---> Series (1 column), DataFrame (several columns), with MultIndex (different layers)

# <center> <img src="img/pandas_df.png" width="700"/> </center>

# In[1]:


import numpy as np
import pandas as pd


# Pandas data structures: the ``Series``, ``DataFrame``, and ``Index``.

# <center> <img src="img/pandas_series_df.png" width="700"/> </center>

# ## The Pandas Series Object
# 
# A Pandas ``Series`` is a one-dimensional array of indexed data.
# It can be created from a list or array as follows:

# In[5]:


data = pd.Series([0.25, 0.5, 0.75, 1.0])
data


# As we see in the output, the ``Series`` wraps both a sequence of values and a sequence of indices, which we can access with the ``values`` and ``index`` attributes.
# The ``values`` are simply a familiar NumPy array:

# In[7]:


data.values


# The ``index`` is an array-like object of type ``pd.Index``, which we'll discuss in more detail momentarily.

# In[8]:


data.index


# Like with a NumPy array, data can be accessed by the associated index via the familiar Python square-bracket notation:

# In[9]:


data[1]


# In[10]:


data[1:3]


# In[15]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data


# # loc, iloc

# In[20]:


data.iloc[0]


# In[22]:


data['a']


# In[23]:


data.loc['a']


# In[24]:


data['b']


# We can even use non-contiguous or non-sequential indices:

# In[34]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data


# In[35]:


data.index


# In[26]:


data[5]


# In[27]:


population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population


# By default, a ``Series`` will be created where the index is drawn from the sorted keys.
# From here, typical dictionary-style item access can be performed:

# In[29]:


population['California']


# In[30]:


population.values


# Unlike a dictionary, though, the ``Series`` also supports array-style operations such as slicing:

# In[31]:


population['California':'Illinois']


# ### Constructing Series objects
# 
# We've already seen a few ways of constructing a Pandas ``Series`` from scratch; all of them are some version of the following:
# 
# ```python
# >>> pd.Series(data, index=index)
# ```
# 
# where ``index`` is an optional argument, and ``data`` can be one of many entities.
# 
# For example, ``data`` can be a list or NumPy array, in which case ``index`` defaults to an integer sequence:

# In[18]:


pd.Series([2, 4, 6])


# ``data`` can be a scalar, which is repeated to fill the specified index:

# In[15]:


pd.Series(5, index=[100, 200, 300])


# ``data`` can be a dictionary, in which ``index`` defaults to the sorted dictionary keys:

# In[16]:


pd.Series({2:'a', 1:'b', 3:'c'})


# In each case, the index can be explicitly set if a different result is preferred:

# ## The Pandas DataFrame Object
# 
# The next fundamental structure in Pandas is the ``DataFrame``.
# Like the ``Series`` object discussed in the previous section, the ``DataFrame`` can be thought of either as a generalization of a NumPy array, or as a specialization of a Python dictionary.
# We'll now take a look at each of these perspectives.

# ### DataFrame as a generalized NumPy array
# If a ``Series`` is an analog of a one-dimensional array with flexible indices, a ``DataFrame`` is an analog of a two-dimensional array with both flexible row indices and flexible column names.
# Just as you might think of a two-dimensional array as an ordered sequence of aligned one-dimensional columns, you can think of a ``DataFrame`` as a sequence of aligned ``Series`` objects.
# Here, by "aligned" we mean that they share the same index.
# 
# To demonstrate this, let's first construct a new ``Series`` listing the area of each of the five states discussed in the previous section:

# In[44]:


area_dict = {'California': 423967, 'Illinois': 149995, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, }
area = pd.Series(area_dict)
area


# In[38]:


population


# Now that we have this along with the ``population`` Series from before, we can use a dictionary to construct a single two-dimensional object containing this information:

# In[45]:


states = pd.DataFrame({'population': population,
                       'area': area})
states


# Like the ``Series`` object, the ``DataFrame`` has an ``index`` attribute that gives access to the index labels:

# In[46]:


states.index


# Additionally, the ``DataFrame`` has a ``columns`` attribute, which is an ``Index`` object holding the column labels:

# In[47]:


states.columns


# Thus the ``DataFrame`` can be thought of as a generalization of a two-dimensional NumPy array, where both the rows and columns have a generalized index for accessing the data.

# ### Constructing DataFrame objects

# #### From a single Series object
# 
# A ``DataFrame`` is a collection of ``Series`` objects, and a single-column ``DataFrame`` can be constructed from a single ``Series``:

# In[21]:


pd.DataFrame(population, columns=['population'])


# #### From a list of dicts
# 
# Any list of dictionaries can be made into a ``DataFrame``.
# We'll use a simple list comprehension to create some data:

# In[51]:


data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
data = pd.DataFrame(data, index=['a', 'b', 'c'])
data


# In[50]:


data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
data


# Even if some keys in the dictionary are missing, Pandas will fill them in with ``NaN`` (i.e., "not a number") values:

# In[60]:


df = pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
df


# In[61]:


df.loc[0, 'c'] = 6


# In[62]:


df


# #### From a dictionary of Series objects
# 
# As we saw before, a ``DataFrame`` can be constructed from a dictionary of ``Series`` objects as well:

# In[28]:


pd.DataFrame({'population': population,
              'area': area, 'area2': area})


# #### From a two-dimensional NumPy array
# 
# Given a two-dimensional array of data, we can create a ``DataFrame`` with any specified column and index names.
# If omitted, an integer index will be used for each:

# In[66]:


data = pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
                   index=[1, 2, 5])
data


# In[65]:


np.random.rand(3, 2)


# ## The Pandas Index Object
# 
# We have seen here that both the ``Series`` and ``DataFrame`` objects contain an explicit *index* that lets you reference and modify data.
# This ``Index`` object is an interesting structure in itself, and it can be thought of either as an *immutable array* or as an *ordered set* (technically a multi-set, as ``Index`` objects may contain repeated values).
# Those views have some interesting consequences in the operations available on ``Index`` objects.
# As a simple example, let's construct an ``Index`` from a list of integers:

# In[30]:


ind = pd.Index([2, 3, 5, 7, 11])
ind


# In[27]:


print(ind.size, ind.shape, ind.ndim, ind.dtype)


# One difference between ``Index`` objects and NumPy arrays is that indices are immutable–that is, they cannot be modified via the normal means:
# 
# 
# ind[1] = 0

# ## Data Selection in DataFrame
# 
# Recall that a ``DataFrame`` acts in many ways like a two-dimensional or structured array, and in other ways like a dictionary of ``Series`` structures sharing the same index.
# 
# loc, iloc, 

# First, the ``loc`` attribute allows indexing and slicing that always references the explicit index:

# In[67]:


data


# In[76]:


data.loc[1]


# In[78]:


data.loc[:, 'foo']


# In[50]:


data.iloc[0].loc['foo']


# In[79]:


data['foo']


# In[81]:


data.bar


# col = 'a' (position 2)
# 
# df.a, df['a'], df.loc[:,'a'], df.iloc[:, 2]

# In[53]:


data.loc[1:4]


# The iloc attribute allows indexing and slicing that always references the implicit Python-style index:

# In[30]:


data.iloc[1]


# In[31]:


data.iloc[1:3]


# ## Operating on Null Values
# 
# As we have seen, Pandas treats ``None`` and ``NaN`` as essentially interchangeable for indicating missing or null values.
# To facilitate this convention, there are several useful methods for detecting, removing, and replacing null values in Pandas data structures.
# They are:
# 
# - ``isnull()``: Generate a boolean mask indicating missing values
# - ``notnull()``: Opposite of ``isnull()``
# - ``dropna()``: Return a filtered version of the data
# - ``fillna()``: Return a copy of the data with missing values filled or imputed

# In[100]:


data = pd.Series([1, np.nan, 'hello', None])
data


# In[88]:


mask = data.notnull()


# In[90]:


mask.index


# In[55]:


data


# In[91]:


data[mask]


# In[101]:


data2 = data.dropna() # inplace = True


# In[103]:


data2


# In[60]:


data2 = data.dropna()
data2


# In[61]:


data


# In[62]:


data.dropna(inplace=True)
data


# In[110]:


df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      6],
                   [np.nan, 4,      6]], index=['a', 'b', 'c'])
df


# We cannot drop single values from a DataFrame; we can only drop full rows or full columns. Depending on the application, you might want one or the other, so dropna() gives a number of options for a DataFrame.
# By default, dropna() will drop all rows in which any null value is present:

# In[111]:


df.dropna()


# In[114]:


df.dropna(axis='columns', how='all')


# But this drops some good data as well; you might rather be interested in dropping rows or columns with *all* NA values, or a majority of NA values.
# This can be specified through the ``how`` or ``thresh`` parameters, which allow fine control of the number of nulls to allow through.
# 
# The default is ``how='any'``, such that any row or column (depending on the ``axis`` keyword) containing a null value will be dropped.
# You can also specify ``how='all'``, which will only drop rows/columns that are *all* null values:

# In[67]:


df[3] = np.nan
df


# In[68]:


df.dropna(axis='columns', how='all')


# For finer-grained control, the ``thresh`` parameter lets you specify a minimum number of non-null values for the row/column to be kept:

# In[115]:


df.dropna(axis='rows', thresh=3)


# We can fill NA entries with a single value, such as zero:
# 

# In[118]:


df.fillna(-999)


# In[123]:


data.fillna(method='bfill').fillna(897)


# In[121]:


data


# In[4]:


df.fillna(method='ffill', axis=1)


# ## Merging, concat

# In[ ]:





# ## Multi-Index
# 

# <center> <img src="img/pandas_df_multindex.png" width="500"/> </center>

# In[ ]:


## Data Selection in DataFrame

Recall that a ``DataFrame`` acts in many ways like a two-dimensional or structured array, and in other ways like a dictionary of ``Series`` structures sharing the same index.

loc, iloc, 

