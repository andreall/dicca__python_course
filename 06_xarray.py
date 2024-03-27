# %% [markdown]
# ![xarray Logo](http://xarray.pydata.org/en/stable/_static/dataset-diagram-logo.png "xarray Logo")
# 
# # Introduction to Xarray

# %% [markdown]
# ---

# %%
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

# %% [markdown]
# ## Introducing the `DataArray` and `Dataset`
# 
# Xarray expands on the capabilities on NumPy arrays, providing a lot of streamlined data manipulation. It is similar in that respect to Pandas, but whereas Pandas excels at working with tabular data, Xarray is focused on N-dimensional arrays of data (i.e. grids). Its interface is based largely on the netCDF data model (variables, attributes, and dimensions), but it goes beyond the traditional netCDF interfaces to provide functionality similar to netCDF-java's [Common Data Model (CDM)](https://docs.unidata.ucar.edu/netcdf-java/current/userguide/common_data_model_overview.html). 

# %% [markdown]
# <center> <img src="img/dataset-diagram.png" width="700"/> </center>

# %% [markdown]
# ### Creation of a `DataArray` object
# 
# The `DataArray` is one of the basic building blocks of Xarray (see docs [here](http://xarray.pydata.org/en/stable/user-guide/data-structures.html#dataarray)). It provides a `numpy.ndarray`-like object that expands to provide two critical pieces of functionality:
# 
# 1. Coordinate names and values are stored with the data, making slicing and indexing much more powerful
# 2. It has a built-in container for attributes
# 
# Here we'll initialize a `DataArray` object by wrapping a plain NumPy array, and explore a few of its properties.

# %% [markdown]
# #### Generate a random numpy array
# 
# For our first example, we'll just create a random array of "temperature" data in units of Kelvin:

# %%
data = 283 + 5 * np.random.randn(5, 3, 4)
data

# %% [markdown]
# #### Wrap the array: first attempt
# 
# Now we create a basic `DataArray` just by passing our plain `data` as input:

# %%
temp = xr.DataArray(data)
temp

# %% [markdown]
# Note two things:
# 
# 1. Xarray generates some basic dimension names for us (`dim_0`, `dim_1`, `dim_2`). We'll improve this with better names in the next example.
# 2. Wrapping the numpy array in a `DataArray` gives us a rich display in the notebook! (Try clicking the array symbol to expand or collapse the view)

# %% [markdown]
# #### Assign dimension names
# 
# Much of the power of Xarray comes from making use of named dimensions. So let's add some more useful names! We can do that by passing an ordered list of names using the keyword argument `dims`:

# %%
temp = xr.DataArray(data, dims=['time', 'lat', 'lon'])
temp.lat

# %% [markdown]
# This is already improved upon from a NumPy array, because we have names for each of the dimensions (or axes in NumPy parlance). Even better, we can take arrays representing the values for the coordinates for each of these dimensions and associate them with the data when we create the `DataArray`. We'll see this in the next example.

# %% [markdown]
# ### Create a `DataArray` with named Coordinates
# 
# #### Make time and space coordinates
# 
# Here we will use [Pandas](../pandas) to create an array of [datetime data](../datetime), which we will then use to create a `DataArray` with a named coordinate `time`.

# %%
times = pd.date_range('2018-01-01', periods=5)
times

# %% [markdown]
# We'll also create arrays to represent sample longitude and latitude:

# %%
lons = np.linspace(-120, -60, 4)
lats = np.linspace(25, 55, 3)

# %% [markdown]
# #### Initialize the `DataArray` with complete coordinate info
# 
# When we create the `DataArray` instance, we pass in the arrays we just created:

# %%
temp = xr.DataArray(data, 
                    coords=[times, lats, lons], 
                    dims=['time', 'lat', 'lon'])
temp

# %% [markdown]
# #### Set useful attributes
# 
# ...and while we're at it, we can also set some attribute metadata:

# %%
temp.attrs['units'] = 'kelvin'
temp.attrs['standard_name'] = 'air_temperature'

temp

# %% [markdown]
# #### Attributes are not preserved by default!
# 
# Notice what happens if we perform a mathematical operaton with the `DataArray`: the coordinate values persist, but the attributes are lost. This is done because it is very challenging to know if the attribute metadata is still correct or appropriate after arbitrary arithmetic operations.
# 
# To illustrate this, we'll do a simple unit conversion from Kelvin to Celsius:

# %%
temp_in_celsius = temp - 273.15
temp_in_celsius

# %% [markdown]
# For an in-depth discussion of how Xarray handles metadata, start in the Xarray docs [here](http://xarray.pydata.org/en/stable/getting-started-guide/faq.html#approach-to-metadata).

# %% [markdown]
# ### The `Dataset`: a container for `DataArray`s with shared coordinates
# 
# Along with `DataArray`, the other key object type in Xarray is the `Dataset`: a dictionary-like container that holds one or more `DataArray`s, which can also optionally share coordinates (see docs [here](http://xarray.pydata.org/en/stable/user-guide/data-structures.html#dataset)).
# 
# The most common way to create a `Dataset` object is to load data from a file (see [below](#Opening-netCDF-data)). Here, instead, we will create another `DataArray` and combine it with our `temp` data.
# 
# This will illustrate how the information about common coordinate axes is used.

# %% [markdown]
# #### Create a pressure `DataArray` using the same coordinates
# 
# This code mirrors how we created the `temp` object above.

# %%
pressure_data = 1000.0 + 5 * np.random.randn(5, 3, 4)
pressure = xr.DataArray(
    pressure_data, coords=[times, lats, lons], dims=['time', 'lat', 'lon']
)
pressure.attrs['units'] = 'hPa'
pressure.attrs['standard_name'] = 'air_pressure'

pressure

# %% [markdown]
# #### Create a `Dataset` object
# 
# Each `DataArray` in our `Dataset` needs a name! 
# 
# The most straightforward way to create a `Dataset` with our `temp` and `pressure` arrays is to pass a dictionary using the keyword argument `data_vars`:

# %%
ds = xr.Dataset(data_vars={'Temperature': temp, 'Pressure': pressure})
ds

# %% [markdown]
# Notice that the `Dataset` object `ds` is aware that both data arrays sit on the same coordinate axes.

# %% [markdown]
# #### Access Data variables and Coordinates in a `Dataset`
# 
# We can pull out any of the individual `DataArray` objects in a few different ways.
# 
# Using the "dot" notation:

# %%
ds.Pressure

# %% [markdown]
# ... or using dictionary access like this:

# %%
ds['Pressure']

# %% [markdown]
# We'll return to the `Dataset` object when we start loading data from files.

# %% [markdown]
# ## Subsetting and selection by coordinate values
# 
# Much of the power of labeled coordinates comes from the ability to select data based on coordinate names and values, rather than array indices. We'll explore this briefly here.
# 
# ### NumPy-like selection
# 
# Suppose we want to extract all the spatial data for one single date: January 2, 2018. It's possible to achieve that with NumPy-like index selection:

# %%
indexed_selection = temp[1, :, :]  # Index 1 along axis 0 is the time slice we want...
indexed_selection

# %% [markdown]
# HOWEVER, notice that this requires us (the user / programmer) to have **detailed knowledge** of the order of the axes and the meaning of the indices along those axes!
# 
# _**Named coordinates free us from this burden...**_

# %% [markdown]
# ### Selecting with `.sel()`
# 
# We can instead select data based on coordinate values using the `.sel()` method, which takes one or more named coordinate(s) as keyword argument:

# %%
named_selection = temp.sel(time='2018-01-02')
named_selection

# %% [markdown]
# We got the same result, but 
# - we didn't have to know anything about how the array was created or stored
# - our code is agnostic about how many dimensions we are dealing with
# - the intended meaning of our code is much clearer!

# %% [markdown]
# ### Approximate selection and interpolation
# 
# With time and space data, we frequently want to sample "near" the coordinate points in our dataset. Here are a few simple ways to achieve that.
# 
# #### Nearest-neighbor sampling
# 
# Suppose we want to sample the nearest datapoint within 2 days of date `2018-01-07`. Since the last day on our `time` axis is `2018-01-05`, this is well-posed.
# 
# `.sel` has the flexibility to perform nearest neighbor sampling, taking an optional tolerance:

# %%
temp.sel(time='2018-01-07', method='nearest', tolerance=timedelta(days=2))

# %% [markdown]
# where we see that `.sel` indeed pulled out the data for date `2018-01-05`.

# %% [markdown]
# #### Interpolation
# 
# The `.interp()` method (see the docs [here](http://xarray.pydata.org/en/stable/interpolation.html)) works similarly to `.sel()`. Using `.interp()`, we can interpolate to any latitude/longitude location:

# %% [markdown]
# 

# %%
temp.interp(lon=-105, lat=40)

# %% [markdown]
# <div class="admonition alert alert-info">
#     <p class="admonition-title" style="font-weight:bold">Info</p>
#     Xarray's interpolation functionality requires the <a href="https://scipy.org/">SciPy</a> package!
# </div>

# %% [markdown]
# ### Slicing along coordinates
# 
# Frequently we want to select a range (or _slice_) along one or more coordinate(s). We can achieve this by passing a Python [slice](https://docs.python.org/3/library/functions.html#slice) object to `.sel()`, as follows:

# %%
temp.sel(
    time=slice('2018-01-01', '2018-01-03'), lon=slice(-110, -70), lat=slice(25, 45)
)

# %% [markdown]
# <div class="admonition alert alert-info">
#     <p class="admonition-title" style="font-weight:bold">Info</p>
#     The calling sequence for <code>slice</code> always looks like <code>slice(start, stop[, step])</code>, where <code>step</code> is optional.
# </div>

# %% [markdown]
# Notice how the length of each coordinate axis has changed due to our slicing.

# %% [markdown]
# ### One more selection method: `.loc`
# 
# All of these operations can also be done within square brackets on the `.loc` attribute of the `DataArray`:
# 

# %%
temp.loc['2018-01-02']

# %% [markdown]
# This is sort of in between the NumPy-style selection
# ```
# temp[1,:,:]
# ```
# and the fully label-based selection using `.sel()`
# 
# With `.loc`, we make use of the coordinate *values*, but lose the ability to specify the *names* of the various dimensions. Instead, the slicing must be done in the correct order:

# %%
temp.loc['2018-01-01':'2018-01-03', 25:45, -110:-70]

# %% [markdown]
# One advantage of using `.loc` is that we can use NumPy-style slice notation like `25:45`, rather than the more verbose `slice(25,45)`. But of course that also works:

# %%
temp.loc['2018-01-01':'2018-01-03', slice(25, 45), -110:-70]

# %% [markdown]
# What *doesn't* work is passing the slices in a different order:

# %%
# This will generate an error
# temp.loc[-110:-70, 25:45,'2018-01-01':'2018-01-03']

# %% [markdown]
# # Opening data
# 
# With its close ties to the netCDF data model, Xarray also supports netCDF as a first-class file format. This means it has easy support for opening netCDF datasets, so long as they conform to some of Xarray's limitations (such as 1-dimensional coordinates).
# 
# ### Access netCDF data with `xr.open_dataset`

# %% [markdown]
# <div class="admonition alert alert-info">
#     <p class="admonition-title" style="font-weight:bold">Info</p>
#     Here we're getting the data from Project Pythia's custom library of example data, which we already imported above with <code>from pythia_datasets import DATASETS</code>. The <code>DATASETS.fetch()</code> method will automatically download and cache our example data file <code>NARR_19930313_0000.nc</code> locally.
# </div>

# %%
#filepath = DATASETS.fetch('NARR_19930313_0000.nc')

# %% [markdown]
# Once we have a valid path to a data file that Xarray knows how to read, we can open it like this:

# %% [markdown]
# ### Open / Load 
# 
# currents_liguria --> folder
# currents_01012010.nc
# currents_02012010.nc
# currents_03012010.nc
# 
# dir_data = Path('currents_liguria')
# 
# xr.open_dataset(dir_data / 'currents_01012010.nc')  
# xr.load_dataset(dir_data / 'currents_01012010.nc')
# 
# xr.open_dataarray(dir_data / 'uo_currents_01012010.nc')
# 
# xr.open_mfdataset(dir_data.glob('currents_*.nc'))
# xr.open_mfdataset(dir_data.glob('currents_*2010.nc'))
# 
# xr.open_mfdataset('XXX*.grb2', engine='cfgrib')
# 
# 

# %%
ds = xr.open_mfdataset('data/WW3_mediterr_*.grb2', engine='cfgrib')
ds

# %% [markdown]
# ## Subsetting the `Dataset`
# 
# Our call to `xr.open_dataset()` above returned a `Dataset` object that we've decided to call `ds`. We can then pull out individual fields:

# %%
ds = xr.open_dataset('data/NARR_19930313_0000.nc')

ds.isobaric1

# %% [markdown]
# (recall that we can also use dictionary syntax like `ds['isobaric1']` to do the same thing)

# %% [markdown]
# `Dataset`s also support much of the same subsetting operations as `DataArray`, but will perform the operation on all data:

# %%
ds_1000 = ds.sel(isobaric1=1000.0)
ds_1000

# %% [markdown]
# And further subsetting to a single `DataArray`:

# %%
ds_1000.Temperature_isobaric

# %% [markdown]
# ### Aggregation operations
# 
# Not only can you use the named dimensions for manual slicing and indexing of data, but you can also use it to control aggregation operations, like `std` (standard deviation):

# %%
u_winds = ds['u-component_of_wind_isobaric']
u_winds.std(dim=['x', 'y'])

# %% [markdown]
# <div class="admonition alert alert-info">
#     <p class="admonition-title" style="font-weight:bold">Info</p>
#     Aggregation methods for Xarray objects operate over the named coordinate dimension(s) specified by keyword argument <code>dim</code>. Compare to NumPy, where aggregations operate over specified numbered <code>axes</code>.
# </div>

# %% [markdown]
# Using the sample dataset, we can calculate the mean temperature profile (temperature as a function of pressure) over Colorado within this dataset. For this exercise, consider the bounds of Colorado to be:
#  * x: -182km to 424km
#  * y: -1450km to -990km
#     
# (37°N to 41°N and 102°W to 109°W projected to Lambert Conformal projection coordinates)

# %%
temps = ds.Temperature_isobaric
co_temps = temps.sel(x=slice(-182, 424), y=slice(-1450, -990))
prof = co_temps.mean(dim=['x', 'y'])
prof

# %% [markdown]
# ## Plotting with Xarray
# 
# Another major benefit of using labeled data structures is that they enable automated plotting with sensible axis labels. 
# 
# ### Simple visualization with `.plot()`
# 
# Much like we saw in [Pandas](../pandas/pandas), Xarray includes an interface to [Matplotlib](../matplotlib) that we can access through the `.plot()` method of every `DataArray`.
# 
# For quick and easy data exploration, we can just call `.plot()` without any modifiers:

# %%
prof.plot()

# %% [markdown]
# Here Xarray has generated a line plot of the temperature data against the coordinate variable `isobaric`. Also the metadata are used to auto-generate axis labels and units.

# %% [markdown]
# ### Customizing the plot
# 
# As in Pandas, the `.plot()` method is mostly just a wrapper to Matplotlib, so we can customize our plot in familiar ways.
# 
# In this air temperature profile example, we would like to make two changes:
# - swap the axes so that we have isobaric levels on the y (vertical) axis of the figure
# - make pressure decrease upward in the figure, so that up is up
# 
# A few keyword arguments to our `.plot()` call will take care of this:

# %%
prof.plot(y="isobaric1", yincrease=False)

# %% [markdown]
# ### Plotting 2D data
# 
# In the example above, the `.plot()` method produced a line plot.
# 
# What if we call `.plot()` on a 2D array?

# %%
temps.sel(isobaric1=1000).plot()

# %% [markdown]
# Xarray has recognized that the `DataArray` object calling the plot method has two coordinate variables, and generates a 2D plot using the `pcolormesh` method from Matplotlib.
# 
# In this case, we are looking at air temperatures on the 1000 hPa isobaric surface over North America. We could of course improve this figure by using [Cartopy](../cartopy) to handle the map projection and geographic features!

# %% [markdown]
# ---


