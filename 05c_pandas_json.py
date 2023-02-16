#!/usr/bin/env python
# coding: utf-8

# ## JSON files
# 
# A JSON file is a file that stores simple data structures and objects in JavaScript Object Notation (JSON) format, which is a standard data interchange format. It is primarily used for transmitting data between a web application and a server. JSON files are lightweight, text-based, human-readable, and can be edited using a text editor.

# In[1]:


import pandas as pd
from pathlib import Path


# In[2]:


dir_file = Path('__file__').resolve().parents[0]


# In[13]:


dir_data = dir_file / 'data' / 'Storico meteo'

year = 2012
fns = dir_data.glob(f'meteo-{year}-*.json')
ff = sorted(fns)


# In[14]:


# DO NOT RUN THIS CELL
%%time
df = pd.read_json(ff[0], lines=True)


# In[15]:


df


# In[16]:


get_ipython().run_cell_magic('time', '', "df = pd.read_json(ff[0], lines=True, chunksize=10000) # chunksize is the number of rows per chunk\ndf_list = list()\nfor c in df:\n    c.drop(columns=['version', 'ident', 'network'], axis=1, inplace=True)\n    c.set_index('date', inplace=True)\n    #pd.json_normalize(c.data.values[0])\n    value_speed = [x[0]['vars']['B05001']['v'] for x in c.data.values if 'B05001' in x[0]['vars']]\n    c['w_speed'] = value_speed\n    c.drop(['data'], axis=1, inplace=True)\n    df_list.append(c)\ndf = pd.concat(df_list)")


# ### Dask
# 
# 
# Parallelize any Python code with Dask Futures, letting you scale any function and for loop, and giving you control and power in any situation.

# In[32]:


import dask.dataframe as dd


# In[128]:


get_ipython().run_cell_magic('time', '', "ddf = dd.read_json(ff[0], blocksize=5000000) # blocksize is size in bytes of each block  \nddf = ddf.drop(columns=['version', 'ident', 'network'], axis=1)\nlist_var = list()\nfor np in range(ddf.npartitions):\n    ddfp = ddf.partitions[np]\n    ddfp = ddfp.set_index('date')\n    value_speed = [x[0]['vars']['B05001']['v'] for x in ddfp.data.compute().values]\n    list_var.append(value_speed)\n#ddf = ddf.drop(columns=['data'], axis=1)")


# #### Parallelization!
