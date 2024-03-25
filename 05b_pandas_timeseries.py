# %% [markdown]
# # Pandas: Time Series

# %% [markdown]
# ### Native Python dates and times: ``datetime`` and ``dateutil``
# 
# Python's basic objects for working with dates and times reside in the built-in ``datetime`` module.
# Along with the third-party ``dateutil`` module, you can use it to quickly perform a host of useful functionalities on dates and times.
# For example, you can manually build a date using the ``datetime`` type:

# %%
from datetime import datetime
datetime(year=2015, month=7, day=4)

# %% [markdown]
# Or, using the ``dateutil`` module, you can parse dates from a variety of string formats:

# %%
from dateutil import parser
date = parser.parse("4th of July, 2015")
date

# %%
date

# %% [markdown]
# Once you have a ``datetime`` object, you can do things like printing the day of the week:

# %%
date.strftime('%A')

# %% [markdown]
# In the final line, we've used one of the standard string format codes for printing dates (``"%A"``), which you can read about in the [strftime section](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) of Python's [datetime documentation](https://docs.python.org/3/library/datetime.html).
# Documentation of other useful date utilities can be found in [dateutil's online documentation](http://labix.org/python-dateutil).
# A related package to be aware of is [``pytz``](http://pytz.sourceforge.net/), which contains tools for working with the most migrane-inducing piece of time series data: time zones.
# 
# The power of ``datetime`` and ``dateutil`` lie in their flexibility and easy syntax: you can use these objects and their built-in methods to easily perform nearly any operation you might be interested in.
# Where they break down is when you wish to work with large arrays of dates and times:
# just as lists of Python numerical variables are suboptimal compared to NumPy-style typed numerical arrays, lists of Python datetime objects are suboptimal compared to typed arrays of encoded dates.

# %% [markdown]
# ### Typed arrays of times: NumPy's ``datetime64``
# 
# The weaknesses of Python's datetime format inspired the NumPy team to add a set of native time series data type to NumPy.
# The ``datetime64`` dtype encodes dates as 64-bit integers, and thus allows arrays of dates to be represented very compactly.
# The ``datetime64`` requires a very specific input format:

# %%
import numpy as np
date = np.array('2015-07-04 05:50', dtype=np.datetime64)
date

# %% [markdown]
# Once we have this date formatted, however, we can quickly do vectorized operations on it:

# %%
date + np.arange(12)

# %%
aa = np.datetime64('2015-07-04 12:00:00')

# %% [markdown]
# Notice that the time zone is automatically set to the local time on the computer executing the code.
# You can force any desired fundamental unit using one of many format codes; for example, here we'll force a nanosecond-based time:

# %%
np.datetime64('2015-07-04 12:59:59.50', 'ns')

# %% [markdown]
# The following table, drawn from the [NumPy datetime64 documentation](http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html), lists the available format codes along with the relative and absolute timespans that they can encode:

# %% [markdown]
# |Code    | Meaning     | Time span (relative) | Time span (absolute)   |
# |--------|-------------|----------------------|------------------------|
# | ``Y``  | Year	       | ± 9.2e18 years       | [9.2e18 BC, 9.2e18 AD] |
# | ``M``  | Month       | ± 7.6e17 years       | [7.6e17 BC, 7.6e17 AD] |
# | ``W``  | Week	       | ± 1.7e17 years       | [1.7e17 BC, 1.7e17 AD] |
# | ``D``  | Day         | ± 2.5e16 years       | [2.5e16 BC, 2.5e16 AD] |
# | ``h``  | Hour        | ± 1.0e15 years       | [1.0e15 BC, 1.0e15 AD] |
# | ``m``  | Minute      | ± 1.7e13 years       | [1.7e13 BC, 1.7e13 AD] |
# | ``s``  | Second      | ± 2.9e12 years       | [ 2.9e9 BC, 2.9e9 AD]  |
# | ``ms`` | Millisecond | ± 2.9e9 years        | [ 2.9e6 BC, 2.9e6 AD]  |
# | ``us`` | Microsecond | ± 2.9e6 years        | [290301 BC, 294241 AD] |
# | ``ns`` | Nanosecond  | ± 292 years          | [ 1678 AD, 2262 AD]    |
# | ``ps`` | Picosecond  | ± 106 days           | [ 1969 AD, 1970 AD]    |
# | ``fs`` | Femtosecond | ± 2.6 hours          | [ 1969 AD, 1970 AD]    |
# | ``as`` | Attosecond  | ± 9.2 seconds        | [ 1969 AD, 1970 AD]    |

# %% [markdown]
# For the types of data we see in the real world, a useful default is ``datetime64[ns]``, as it can encode a useful range of modern dates with a suitably fine precision.

# %% [markdown]
# ### Dates and times in pandas: best of both worlds
# 
# Pandas builds upon all the tools just discussed to provide a ``Timestamp`` object, which combines the ease-of-use of ``datetime`` and ``dateutil`` with the efficient storage and vectorized interface of ``numpy.datetime64``.
# From a group of these ``Timestamp`` objects, Pandas can construct a ``DatetimeIndex`` that can be used to index data in a ``Series`` or ``DataFrame``; we'll see many examples of this below.

# %%
import pandas as pd
# date = pd.to_datetime("4th of July, 2015")
date = pd.to_datetime("04-07-2015 04:30", dayfirst=True)
date

# %%
date.strftime('%A')

# %%
pd.to_timedelta(np.arange(12), 'D')

# %%
date + pd.to_timedelta(np.arange(12), 'D')

# %%
pd.to_timedelta(np.arange(12), 'D')

# %% [markdown]
# ## Pandas Time Series: Indexing by Time
# 
# Where the Pandas time series tools really become useful is when you begin to *index data by timestamps*.
# For example, we can construct a ``Series`` object that has time indexed data:

# %%
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-07', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
data

# %%
data.plot(marker='.')

# %% [markdown]
# Now that we have this data in a ``Series``, we can make use of any of the ``Series`` indexing patterns we discussed in previous sections, passing values that can be coerced into dates:

# %%
data['2014-07-04':'2015-01-01']

# %% [markdown]
# There are additional special date-only indexing operations, such as passing a year to obtain a slice of all data from that year:

# %%
data['2015']

# %%
data.index.hour

# %%
data[data.index.month == 12]

# %%
data[data.index.day==7]

# %% [markdown]
# Later, we will see additional examples of the convenience of dates-as-indices.
# But first, a closer look at the available time series data structures.

# %% [markdown]
# ## Pandas Time Series Data Structures
# 
# This section will introduce the fundamental Pandas data structures for working with time series data:
# 
# - For *time stamps*, Pandas provides the ``Timestamp`` type. As mentioned before, it is essentially a replacement for Python's native ``datetime``, but is based on the more efficient ``numpy.datetime64`` data type. The associated Index structure is ``DatetimeIndex``.
# - For *time Periods*, Pandas provides the ``Period`` type. This encodes a fixed-frequency interval based on ``numpy.datetime64``. The associated index structure is ``PeriodIndex``.
# - For *time deltas* or *durations*, Pandas provides the ``Timedelta`` type. ``Timedelta`` is a more efficient replacement for Python's native ``datetime.timedelta`` type, and is based on ``numpy.timedelta64``. The associated index structure is ``TimedeltaIndex``.

# %% [markdown]
# The most fundamental of these date/time objects are the ``Timestamp`` and ``DatetimeIndex`` objects.
# While these class objects can be invoked directly, it is more common to use the ``pd.to_datetime()`` function, which can parse a wide variety of formats.
# Passing a single date to ``pd.to_datetime()`` yields a ``Timestamp``; passing a series of dates by default yields a ``DatetimeIndex``:

# %%
dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                       '2015-Jul-6', '07-07-2015', '20150708'])
dates

# %% [markdown]
# A ``TimedeltaIndex`` is created, for example, when a date is subtracted from another:

# %%
dates[1:] - dates[0:-1]

# %% [markdown]
# ### Regular sequences: ``pd.date_range()``
# 
# To make the creation of regular date sequences more convenient, Pandas offers a few functions for this purpose: ``pd.date_range()`` for timestamps, ``pd.period_range()`` for periods, and ``pd.timedelta_range()`` for time deltas.
# We've seen that Python's ``range()`` and NumPy's ``np.arange()`` turn a startpoint, endpoint, and optional stepsize into a sequence.
# Similarly, ``pd.date_range()`` accepts a start date, an end date, and an optional frequency code to create a regular sequence of dates.
# By default, the frequency is one day:

# %%
np.arange(0, 10)

# %%
pd.date_range('2015-07-03', '2020-07-03')

# %% [markdown]
# Alternatively, the date range can be specified not with a start and endpoint, but with a startpoint and a number of periods:

# %%
pd.date_range('2015-07-03', periods=8)

# %% [markdown]
# The spacing can be modified by altering the ``freq`` argument, which defaults to ``D``.
# For example, here we will construct a range of hourly timestamps:

# %%
pd.date_range('2015-07-03', periods=8, freq='MS')

# %% [markdown]
# To create regular sequences of ``Period`` or ``Timedelta`` values, the very similar ``pd.period_range()`` and ``pd.timedelta_range()`` functions are useful.
# Here are some monthly periods:

# %%
pd.period_range('2015-07', periods=8, freq='M')

# %% [markdown]
# And a sequence of durations increasing by an hour:

# %%
pd.timedelta_range('1 days', periods=10, freq='H')

# %% [markdown]
# ## Frequencies and Offsets
# 
# Fundamental to these Pandas time series tools is the concept of a frequency or date offset.
# Just as we saw the ``D`` (day) and ``H`` (hour) codes above, we can use such codes to specify any desired frequency spacing.
# The following table summarizes the main codes available:

# %% [markdown]
# | Code   | Description         | Code   | Description          |
# |--------|---------------------|--------|----------------------|
# | ``D``  | Calendar day        | ``B``  | Business day         |
# | ``W``  | Weekly              |        |                      |
# | ``M``  | Month end           | ``BM`` | Business month end   |
# | ``Q``  | Quarter end         | ``BQ`` | Business quarter end |
# | ``A``  | Year end            | ``BA`` | Business year end    |
# | ``H``  | Hours               | ``BH`` | Business hours       |
# | ``T``  | Minutes             |        |                      |
# | ``S``  | Seconds             |        |                      |
# | ``L``  | Milliseonds         |        |                      |
# | ``U``  | Microseconds        |        |                      |
# | ``N``  | nanoseconds         |        |                      |

# %% [markdown]
# The monthly, quarterly, and annual frequencies are all marked at the end of the specified period.
# By adding an ``S`` suffix to any of these, they instead will be marked at the beginning:

# %% [markdown]
# | Code    | Description            || Code    | Description            |
# |---------|------------------------||---------|------------------------|
# | ``MS``  | Month start            ||``BMS``  | Business month start   |
# | ``QS``  | Quarter start          ||``BQS``  | Business quarter start |
# | ``AS``  | Year start             ||``BAS``  | Business year start    |

# %% [markdown]
# Additionally, you can change the month used to mark any quarterly or annual code by adding a three-letter month code as a suffix:
# 
# - ``Q-JAN``, ``BQ-FEB``, ``QS-MAR``, ``BQS-APR``, etc.
# - ``A-JAN``, ``BA-FEB``, ``AS-MAR``, ``BAS-APR``, etc.
# 
# In the same way, the split-point of the weekly frequency can be modified by adding a three-letter weekday code:
# 
# - ``W-SUN``, ``W-MON``, ``W-TUE``, ``W-WED``, etc.
# 
# On top of this, codes can be combined with numbers to specify other frequencies.
# For example, for a frequency of 2 hours 30 minutes, we can combine the hour (``H``) and minute (``T``) codes as follows:

# %%
pd.to_datetime('2022-07-01') + pd.timedelta_range(0, periods=9, freq="2H30T")

# %% [markdown]
# All of these short codes refer to specific instances of Pandas time series offsets, which can be found in the ``pd.tseries.offsets`` module.
# For example, we can create a business day offset directly as follows:

# %%
from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())

# %% [markdown]
# ## Reading data

# %%
df = pd.read_table('data/data_waves.dat')
df

# %%
df = pd.read_table('data/data_waves.dat', header=None, delim_whitespace=True, 
                   names=['YY', 'mm', 'DD', 'time', 'hs', 'tm', 'tp', 'dirm', 'dp', 'spr', 'h', 'lm', 'lp', 
                          'uw', 'vw'],
                  parse_dates=[[0, 1, 2, 3]], index_col=0)
df

# %% [markdown]
# names=['YY', 'mm', 'DD', 'time', 'hs', 'tm', 'tp', 'dirm', 'dp', 'spr', 'h', 'lm', 'lp', 
#                           'uw', 'vw']

# %%
df = pd.read_table('data/data_waves.dat', header=None, delim_whitespace=True, parse_dates=[[0, 1, 2, 3]], 
                   index_col=0,
                   names=['YY', 'mm', 'DD', 'time', 'hs', 'tm', 'tp', 'dirm', 'dp', 'spr', 'h', 'lm', 'lp', 
                          'uw', 'vw'])
df.loc[df['tp'] < 0, 'tp'] = np.NaN
df.loc[df.tp > 20, 'tp'] = np.NaN
df.dropna(inplace=True) # df = df.dropna()
df

# %%
df.describe()

# %%
df.hs[:10000].plot()

# %%
df.max()

# %%
df.sort_values('hs', ascending=False)

# %% [markdown]
# ## Resampling, Shifting, and Windowing

# %%
df['hs']

# %%
df.hs[:100].plot()

# %%
df.rolling('12H').mean().hs[:100].plot()

# %%
dfi = df.iloc[:500]

# %%
import matplotlib.pyplot as plt
dfi.hs.plot(alpha=0.5, style='-.', marker='o', markersize=1)
dfi.hs.resample('24H').mean().plot(style=':', linewidth=2)
dfi.hs.asfreq('24H').plot(style='--');
plt.legend(['input', 'resample', 'asfreq'],
           loc='upper left');

# %%
#df.hs.plot(alpha=0.5, style='-.', marker='o', markersize=1)
df_res = df.hs.resample('A').mean()
df_res.plot(style=':', linewidth=2)

# %%
df.hs.asfreq('Y').plot()

# %% [markdown]
# Notice the difference: at each point, ``resample`` reports the *average of the previous year*, while ``asfreq`` reports the *value at the end of the year*.

# %% [markdown]
# For up-sampling, ``resample()`` and ``asfreq()`` are largely equivalent, though resample has many more options available.
# In this case, the default for both methods is to leave the up-sampled points empty, that is, filled with NA values.
# Just as with the ``pd.fillna()`` function discussed previously, ``asfreq()`` accepts a ``method`` argument to specify how values are imputed.
# Here, we will resample the business day data at a daily frequency (i.e., including weekends):

# %%
df.resample('A').max()

# %%
annual_max = df.groupby(df.index.year).max()
annual_max

# %%
index_hs_max=df.hs.groupby(df.index.year).idxmax()
index_hs_max

# %%
df.groupby(df.index.month).mean().plot();


