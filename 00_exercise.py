# %% [markdown]
# ### Exercise
# 
# We will analyze the similarities between different mouses using the euclidean distance

# %%
import numpy as np

# %%
weight = np.random.uniform(5, 15, (50, 10)) # gr
size = np.random.uniform(5, 20, (50, 10)) # cm
speed = np.random.randint(1, 5, (50, 10)) # m/s  

# %% [markdown]
# Euclidean distance
# 
# ![Screenshot 2024-03-25 at 09.44.22.png](attachment:16468c44-c92f-44fb-83b1-024eaba87104.png)
# 
# 
# So a single mouse if defined by its weight, size and speed and their evolution in time. We have 10 mice and 50 measurements in time.

# %%
def np_normalize_var(var):
    var_normed = (var - np.mean(var)) / np.std(var)

    return var_normed

# %%
weight_normed = np_normalize_var(weight)
size_normed = np_normalize_var(size)
speed_normed = np_normalize_var(speed)

# %% [markdown]
# ### Comparison of M1 with M2

# %%
d = np.sqrt((weight_normed[:, 0] - weight_normed[:, 1])**2 + 
            (size_normed[:, 0] - size_normed[:, 1])**2 +
            (speed_normed[:, 0] - speed_normed[:, 1])**2)
d

# %% [markdown]
# ### Comparison of M1 with the rest of the mice

# %%
weight_normed[:, 0].shape # the first column is an array of 50 values without (row, column) structure

# %%
# Reshape the array to (50, 1)
weight_normed[:, 0][:, np.newaxis].shape

# %%
np.reshape(weight_normed[:, 0], (50, 1)).shape

# %%
(weight_normed[:, 0][:, np.newaxis] - weight_normed[:, 1:]).shape

# %%
dm1 = np.sqrt((weight_normed[:, 0][:, np.newaxis] - weight_normed[:, 1])**2 + 
            (size_normed[:, 0][:, np.newaxis] - size_normed[:, 1])**2 +
            (speed_normed[:, 0][:, np.newaxis] - speed_normed[:, 1])**2)
dm1

# %% [markdown]
# #### Plot 3,2 plots of weights, sizes, and speed of the mice
# 
# 1st row - raw data
# 
# 2nd row - normalized data

# %%
import matplotlib.pyplot as plt

# %%
fig, axs = plt.subplots(2, 3, sharey='row')

# raw data
for ii, var in enumerate([weight, size, speed]):
    p = axs[0, ii].pcolormesh(var, vmin=1, vmax=19)
    plt.colorbar(p)

# normalized data
for ii, var in enumerate([weight_normed, size_normed, speed_normed]):
    p = axs[1, ii].pcolormesh(var, cmap='Reds', vmin=-2, vmax=2)
plt.colorbar(p)


# %%
plt.pcolormesh(dm1)
plt.colorbar(label='Euc-distance for M1')


