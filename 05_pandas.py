#!/usr/bin/env python
# coding: utf-8

# # Pandas

# At the very basic level, Pandas objects can be thought of as enhanced versions of NumPy structured arrays in which the rows and columns are identified with labels rather than simple integer indices.
# 
# https://www.datacamp.com/tutorial/pandas-multi-index
# 

# ### Numpy:
# Data WITHOUT indexes ---> Arrays (1D or many dimensions)

# <center> <img src="img/numpy_arrays.png" width="600"/> </center>

# ### Pandas:
# Data WITH indexes ---> Series (1 column), DataFrame (several columns), with MultIndex (different layers)

# <center> <img src="img/pandas_df.png" width="700"/> </center>

# In[36]:


import numpy as np
import pandas as pd


# Pandas data structures: the ``Series``, ``DataFrame``, and ``Index``.

# <center> <img src="img/pandas_series_df.png" width="700"/> </center>

# ## The Pandas Series Object
# 
# A Pandas ``Series`` is a one-dimensional array of indexed data.
# It can be created from a list or array as follows:

# In[8]:


data = pd.Series([0.25, 0.5, 0.75, 1.0])
data


# In[10]:


datanp = np.array([0.25, 0.5, 0.75, 1.0])
datanp


# As we see in the output, the ``Series`` wraps both a sequence of values and a sequence of indices, which we can access with the ``values`` and ``index`` attributes.
# The ``values`` are simply a familiar NumPy array:

# In[11]:


data.values


# The ``index`` is an array-like object of type ``pd.Index``, which we'll discuss in more detail momentarily.

# In[12]:


data.index


# Like with a NumPy array, data can be accessed by the associated index via the familiar Python square-bracket notation:

# In[13]:


data[1]


# In[14]:


data[1:3]


# In[18]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data


# In[21]:


data.iloc[1] # CHECH data[1]


# # loc, iloc
# 
# ### loc --> access your series/df by index (rows, columns)
# ### iloc --> access your series/df by position

# In[27]:


data.iloc[0]


# In[28]:


data.loc['a']


# In[23]:


data.loc['a']


# In[24]:


data['b']


# We can even use non-contiguous or non-sequential indices:

# In[29]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data


# In[30]:


data.index


# In[26]:


data[5]


# In[32]:


population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population


# By default, a ``Series`` will be created where the index is drawn from the sorted keys.
# From here, typical dictionary-style item access can be performed:

# In[33]:


population['California']


# In[34]:


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

# In[35]:


pd.Series(5, index=[100, 200, 300])


# ``data`` can be a dictionary, in which ``index`` defaults to the sorted dictionary keys:

# In[36]:


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

# In[37]:


area_dict = {'California': 423967, 'Illinois': 149995, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, }
area = pd.Series(area_dict)
area


# In[38]:


population


# Now that we have this along with the ``population`` Series from before, we can use a dictionary to construct a single two-dimensional object containing this information:

# In[39]:


states = pd.DataFrame({'population': population,
                       'area': area})
states


# Like the ``Series`` object, the ``DataFrame`` has an ``index`` attribute that gives access to the index labels:

# In[40]:


states.index


# Additionally, the ``DataFrame`` has a ``columns`` attribute, which is an ``Index`` object holding the column labels:

# In[41]:


states.columns


# Thus the ``DataFrame`` can be thought of as a generalization of a two-dimensional NumPy array, where both the rows and columns have a generalized index for accessing the data.

# ### Constructing DataFrame objects

# #### From a single Series object
# 
# A ``DataFrame`` is a collection of ``Series`` objects, and a single-column ``DataFrame`` can be constructed from a single ``Series``:

# In[42]:


pd.DataFrame(population, columns=['population'])


# #### From a list of dicts
# 
# Any list of dictionaries can be made into a ``DataFrame``.
# We'll use a simple list comprehension to create some data:

# In[44]:


data = [{'a': i, 'b': 2 * i} for i in range(3)]
data = pd.DataFrame(data, index=['aa', 'bb', 'cc'])
data


# Even if some keys in the dictionary are missing, Pandas will fill them in with ``NaN`` (i.e., "not a number") values:

# In[46]:


df = pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}, {'d':1}])
df


# In[52]:


df.loc[:, 'b']


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

# In[53]:


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

# In[56]:


ind = pd.Index([2, 3, 5, 7, 11])
ind


# In[57]:


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

# In[58]:


data


# In[59]:


data.loc[1]


# In[60]:


data.loc[:, 'foo']


# In[62]:


data.iloc[0].loc['foo']


# In[63]:


data['bar']


# In[64]:


data.bar


# ## col = 'a' (position 2)
# 
# ## df.a, df['a'], df.loc[:,'a'], df.iloc[:, 2]

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


# ## Multi-Index
# 

# <center> <img src="img/pandas_df_multindex.png" width="500"/> </center>

# In[3]:


import seaborn as sns
tips = sns.load_dataset("tips")
tips.head()
tips


# In[2]:


sns.set()
sns.pairplot(tips, hue='day');


# In[4]:


# Get mean of smoker/non-smoker groups
df = tips.groupby('smoker').mean()
df


# In[5]:


df


# In[8]:


df.reset_index()


# In[9]:


# Group by two columns
df = tips.groupby(['smoker','time']).mean()
df


# In[10]:


# Check out index
df.index


# In[11]:


tips.groupby(['smoker','time']).size()


# In[13]:


# Swap levels of multi-index
df.swaplevel()


# In[24]:


# Unstack your multi-index
df.xs('Lunch', level='time')


# In[25]:


df.xs('Yes', level='smoker')


# In[32]:


df.loc['Yes', :]


# In[33]:


df.loc['Yes', :].sum()


# In[31]:


df.loc[('Yes', 'Lunch'), :]


# In[15]:


# Unsstack the outer index
df.unstack(level=0)


# ## Merging, concat

# In[37]:


df1 = pd.DataFrame(
       {
           "A": ["A0", "A1", "A2", "A3"],
           "B": ["B0", "B1", "B2", "B3"],
           "C": ["C0", "C1", "C2", "C3"],
           "D": ["D0", "D1", "D2", "D3"],
       },
       index=[0, 1, 2, 3],
   )
   
df2 = pd.DataFrame(
        {
            "A": ["A4", "A5", "A6", "A7"],
            "B": ["B4", "B5", "B6", "B7"],
            "C": ["C4", "C5", "C6", "C7"],
            "D": ["D4", "D5", "D6", "D7"],
        },
        index=[4, 5, 6, 7],
    )
    

df3 = pd.DataFrame(
       {
           "A": ["A8", "A9", "A10", "A11"],
           "B": ["B8", "B9", "B10", "B11"],
           "C": ["C8", "C9", "C10", "C11"],
           "D": ["D8", "D9", "D10", "D11"],
       },
       index=[8, 9, 10, 11],
   )
   

frames = [df1, df2, df3]

result = pd.concat(frames)


# In[38]:


result


# In[40]:


pd.concat(frames, keys=["x", "y", "z"])


# In[42]:


df4 = pd.DataFrame(
      {
          "B": ["B2", "B3", "B6", "B7"],
          "D": ["D2", "D3", "D6", "D7"],
          "F": ["F2", "F3", "F6", "F7"],
      },
      index=[2, 3, 6, 7],
  )
  

result = pd.concat([df1, df4], axis=1)


# In[44]:


result


# ## Exercise
# 
# https://towardsdatascience.com/how-to-use-multiindex-in-pandas-to-level-up-your-analysis-aeac7f451fce

# In[46]:


# load data
df = pd.read_csv('data/WordsByCharacter.csv')


# In[48]:


df


# In[54]:


multi = df.set_index(['Film', 'Chapter', 'Race', 'Character']).sort_index()


# In[55]:


multi


# Which characters speak in the first chapter of “The Fellowship of the Ring”?

# Who are the first three elves to speak in the “The Fellowship of the Ring”?
# 

# How much do Gandalf and Saruman talk in each chapter of “The Two Towers”?

# Which hobbits speak the most in each film and across all three films?

# ## Extra pandas + seaborn

# In[61]:


# Load the penguins dataset
penguins = sns.load_dataset("penguins")


# In[63]:


penguins


# In[65]:


g = sns.jointplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    kind="kde",
)


# In[68]:


# Draw a categorical scatterplot to show each observation
ax = sns.swarmplot(data=penguins, x="body_mass_g", y="sex", hue="species")
ax.set(ylabel="")


# In[70]:


mpg = sns.load_dataset("mpg")

colors = (250, 70, 50), (350, 70, 50)
cmap = sns.blend_palette(colors, input="husl", as_cmap=True)
sns.displot(
    mpg,
    x="displacement", col="origin", hue="model_year",
    kind="ecdf", aspect=.75, linewidth=2, palette=cmap,
)


# For the LOTR exercise, plot the number of words per character, per film, per chapter --> how would you present the data?
