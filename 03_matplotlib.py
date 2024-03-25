# %% [markdown]
# # Introduction to Matplotlib

# %% [markdown]
# The **Matplotlib** package can be used to make scientific-grade plots. You can import it with:

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# If you are using IPython and you want to make interactive plots, you can start up IPython with:
# 
#     ipython --matplotlib
# 
# If you now type a plotting command, an interactive plot will pop up.
# 
# If you use the IPython notebook, add a cell containing:

# %%
%matplotlib inline

# %% [markdown]
# and the plots will appear inside the notebook.

# %% [markdown]
# ## Basic plotting

# %% [markdown]
# The main plotting function is called ``plot``:

# %%
plt.plot([1,2,3,6,4,2,3,4])

# %% [markdown]
# In the above example, we only gave a single list, so it will assume the x values are the indices of the list/array.

# %% [markdown]
# However, we can instead specify the x values:

# %%
plt.plot([3.3, 4.4, 4.5, 6.5], [3., 5., 6., 7.])

# %%
# plt.show()

# %% [markdown]
# Matplotlib can take Numpy arrays, so we can do for example:

# %%
import numpy as np
x = np.linspace(0., 10., 50)
y = np.sin(x)
plt.plot(x, y)

# %% [markdown]
# The ``plot`` function is actually quite complex, and for example can take arguments specifying the type of point, the color of the line, and the width of the line:

# %%
plt.plot(x, y, 'royalblue', linestyle='--', marker='o', linewidth=4) # m, c, lw, ls

# %% [markdown]
# plt.plot(x, y, 'o', color='k')

# %% [markdown]
# The line can be hidden with:

# %%
plt.figure()
plt.plot(x, y,linewidth=0, color='green',  marker='+')
plt.plot(x+1, y,linewidth=0, color='red',  marker='o')

# %% [markdown]
# If you are interested, you can specify some of these attributes with a special syntax, which you can read up more about in the Matplotlib documentation:

# %%
plt.plot(x, y, 'go')  # means green and circles

# %%
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)

#plt.plot(X, C)

plt.figure()
plt.plot(X, S)

fig = plt.figure()
ax = fig.axes

fig, ax = plt.subplots() #---> let's you define many subplots
ax.plot(X, S) # instead of plt.plot(), you do ax.plot()

# %% [markdown]
# ## Customizing plots

# %% [markdown]
# ### fig, axs = plt.subplots(nrows, ncols, figsize=(8, 6), .....)
# 
# e.g. 
# 
# fig, axs = plt.subplots(2, 3)
# 
# 1, 1 --> axs[0, 0]
# 
# 1, 2 --> axs[0, 1]
# 
# 1, 3 --> axs[0, 2]
# 
# axes = fig.axes() # [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
# 
# 1, 1 --> axes[0]
# 
# 1, 2 --> axes[1]
# 
# 1, 3 --> axes[2]
# 
# 2, 1 --> axes[3]
# 
# ### fig = plt.figure()
# 
# fig.subplot(nrows, ncols, pos)
# 
# e.g.
# 
# plt.figure()
# 
# 1, 1 --> plt.subplot(2, 3, 1)
# 
# plt.plot()
# 
# plt.legend()
# 
# plt.xlim()
# 
# 1, 2 --> plt.subplot(2, 3, 2)
# 

# %%
# Create a figure of size 8x6 inches, 80 dots per inch
plt.figure(figsize=(8, 6), dpi=80)

# Create a new subplot from a grid of 1x2
# plt.subplot(1, 2, 1)

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)

# Plot cosine with a blue continuous line of width 1 (pixels)
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-", label='cosine')

# Plot sine with a green continuous line of width 1 (pixels)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-", label='sine')

# Set x limits
plt.xlim(-4.0, 4) # plt.xlim(-4)

# Set x ticks
plt.xticks(np.linspace(-4, 4, 9), ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b'])

# Set y limits
plt.ylim(-1.0, 1.0)

# Set y ticks
plt.yticks(np.linspace(-1, 1, 5))


plt.xlabel('x values')
plt.ylabel('y values')

# plt.subplot(1, 2, 2)

# ax.set_xlabel()
# axes[0].set_xlabel()
# axs[0, 1].set_xlabel()


# %%
plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-", label="sine")

# plt.legend(loc='upper left', ncol=2, frameon=True, title='Trig') # upper center lower / left right

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True) # upper center lower / left right


# %%
h1 = plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
h2 = plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-")

plt.legend(['cosine', 'sine'], loc='lower center', ncol=2, frameon=False) # upper center lower / left right

# %% [markdown]
# ## Exercise
# 
# Start off by loading the ``data/SIMAR_gaps.txt`` (Numpy lecture):
# 1. Plot Hm0 (complete series)
# 2. Plot Hm0 and, on top, plot the markers of the annual maximum values

# %%
import numpy as np
data = np.loadtxt('data/SIMAR_gaps.txt', skiprows=1)
data[data<0] = np.NaN

# %%
yy = data[:, 0]
mm = data[:, 1]
dd = data[:, 2]
hs = data[:, 4]

# %%
plt.plot(hs)
plt.ylabel('Hs (m)')

# %%
hsmax = np.nanmax(hs)
posmax1 = hs == hsmax

# %%
pos = np.arange(len(hs))

# %%
plt.plot(hs)
# plt.plot(pos[posmax1], hs[posmax1], '*', ms=10)
plt.plot(np.nanargmax(hs), np.nanmax(hs), '+', ms=10)
plt.ylabel('Hs (m)')

# %%
# Remove NaNs 
plt.plot(hs)
hs[np.isnan(hs)] = 0
plt.plot(hs.argmax(), hs.max(), 'go')
# hs.argmax is the same as np.where(hs == np.max(hs))

# %%
plt.figure(figsize=(20, 10))
plt.plot(hs)

mask = hs == np.nanmax(hs)
ind = int(np.where(hs == np.nanmax(hs))[0])
plt.plot(ind, hs[mask], 'go')

# %% [markdown]
# ## Other types of plots

# %% [markdown]
# ### Scatter plots

# %% [markdown]
# While the ``plot`` function can be used to show scatter plots, it is mainly used for line plots, and the ``scatter`` function is more often used for scatter plots, because it allows more fine control of the markers:

# %%
x = np.random.random(100)
y = np.random.random(100)
plt.scatter(x, y)
# ax.scatter()

# %%
plt.scatter(data[:,4], data[:, 5])

# %%


# %% [markdown]
# ### Errorbar

# %%
###  generate some random data
xdata2 = np.arange(15)
ydata2 = np.random.randn(15)
yerrors = np.random.uniform(0, 1, 15)

###  initialize the figure
fig, ax = plt.subplots()

ax.errorbar(xdata2, ydata2, yerr=yerrors)#, xerr=yerrors)
ax.grid()
#ax.grid(color='royalblue', linewidth=5)

# %%
# linestyle = ls; color=c, marker=m
plt.errorbar(xdata2, ydata2, yerr=yerrors, ls='',         # no lines connecting points
             elinewidth=2,  # error line width
                                               ecolor='gray', # error color
                                               marker='*',    # circular plot symbols
                                               ms=20,         # markersize
                                               mfc='g',       # marker face color
                                               mew=2,         # marker edge width
                                               mec='k',       # marker edge color
                                
                                               capsize=6)     # error hat sizex

# %% [markdown]
# ### Histograms

# %% [markdown]
# Histograms are easy to plot using the ``hist`` function:

# %%
np.random.uniform(0, 600, 500)

# %%
v = np.random.uniform(0, 600., 500)
h = plt.hist(v, bins='auto')  
# we do h= to capture the output of the function, but we don't use it
# bins = 'auto' (default), array, list (np.linspace(x, y, nn), int (number of bins))

# %%
h = plt.hist(v, bins=5)  

# %%
np.histogram(v, bins=np.linspace(0, 600, 20))

# %%
plt.hist(v, bins='auto', density=True, cumulative=True)  

# %%
h[1]

# %%
h = plt.hist(v, range=[-5., 15.], bins=100)

# %%
h = plt.hist(v, orientation='horizontal')

# %% [markdown]
# ### Images

# %% [markdown]
# You can also show two-dimensional arrays with the ``imshow`` function:

# %%
array = np.random.random((64, 64))
plt.imshow(array, cmap='viridis') # pcolormesh
plt.colorbar()

# %% [markdown]
# And the colormap can be changed:

# %%
import cmocean as cmo
plt.imshow(array, cmap=cmo.cm.haline)
plt.colorbar(label='hola')
plt.xlabel('var 1')
plt.ylabel('var 2')


# %% [markdown]
# ### Contour

# %%
def f(x,y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y) 

z = f(x, y)
#print(z.size)

Z = f(X, Y)
#print(Z.shape)

plt.axes([0.025, 0.025, 0.95, 0.95])

p = plt.contourf(X, Y, f(X, Y), 8, alpha=.5, cmap=cmo.cm.deep)
C = plt.contour(X, Y, f(X, Y), 8,  colors='k', linewidth=.5)
plt.clabel(C, inline=1, fontsize=15)
plt.colorbar(p)

#plt.xticks([])
#plt.yticks([])
#plt.show()

# %%
Z

# %% [markdown]
# ### Polar plots

# %%
ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)

N = 20
theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
radii = 10 * np.random.rand(N)
width = np.pi / 4* np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)

for r,bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r/10.))
    bar.set_alpha(0.5)


ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()

# %% [markdown]
# ### Multiplots

# %%
# First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# %%
# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True) # sharey = True, 'rows', None
# f, axs = plt.subplots(1, 2, sharey=True)
# axs[0, 0], axs[0, 1]
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y+10)

# %%
# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# %%
# Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')

# Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')

# Share both X and Y axes with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')

# Note that this is the same as
plt.subplots(2, 2, sharex=True, sharey=True)

# %%
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot([1, 2, 3])
axs[0, 1].plot([1, 2, 3])
axs[1, 0].plot([1, 2, 3])
axs[1, 1].remove()

# %%
# GridSpec

# %% [markdown]
# ## Saving plots to files

# %% [markdown]
# To save a plot to a file, you can do for example:

# %%
from pathlib import Path
dir_pics = Path('img')
plt.savefig('my_plot1.png', bbox_inches='tight', dpi=300)
fig.savefig(dir_pics / 'my_plot.png')
fig.savefig('img/myplot.png')
# pdf, png, eps, 

# %% [markdown]
# and you can then view the resulting file like you would iew a normal image. On Linux, you can also do:
# 
#     $ xv my_plot.png
# 
# in the terminal.

# %% [markdown]
# ## Learning more

# %% [markdown]
# The easiest way to find out more about a function and available options is to use the ``?`` help in IPython:
# 
#         In [11]: plt.hist?
# 
#     Definition: plt.hist(x, bins=10, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, **kwargs)
#     Docstring:
#     Plot a histogram.
# 
#     Call signature::
# 
#       hist(x, bins=10, range=None, normed=False, weights=None,
#              cumulative=False, bottom=None, histtype='bar', align='mid',
#              orientation='vertical', rwidth=None, log=False,
#              color=None, label=None, stacked=False,
#              **kwargs)
# 
#     Compute and draw the histogram of *x*. The return value is a
#     tuple (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*,
#     [*patches0*, *patches1*,...]) if the input contains multiple
#     data.
# 
#     etc.
# 
# But sometimes you don't even know how to make a specific type of plot, in which case you can look at the [Matplotlib Gallery](http://matplotlib.org/gallery.html) for example plots and scripts.
# 

# %% [markdown]
# ## Exercise

# %% [markdown]
# 1. Make a figure of two subplots. 
# 2. On the first subplot you can plot Hm0.
# 3. On the second one, plot the histogram of Hm0. Try changing the number of bins and try plotting the CDF on top of it (with a line not bars).

# %%
plt.hist?

# %%
fig, axs = plt.subplots(1, 2)
axs[0].plot(data[:, 4])
axs[0].set_ylabel('Hm0 (m)')
axs[0].grid()
axs[1].hist(data[:, 4], bins=10, density=True)
axs[1].hist(data[:, 4], bins='auto', density=True, cumulative=True, histtype='step', lw=3)
axs[1].grid()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
# data = hs
fig, (ax1, ax2) = plt.subplots(1, 2)
h = ax1.hist(hs, bins='auto')
h = ax2.hist(hs, bins='auto', density='True', cumulative='True')

# %%
#fig, (ax1, ax2) = plt.subplots(1, 2)
#h = ax1.hist(hs, bins='auto', density=True)
h1 = plt.hist(hs, bins='auto', density=True, cumulative=True)
#index = h1[1] + np.diff(h1[1])
#ax1.plot(h1[1], h1[0], '*', color='red')

# %%
Y = h1[0]
bins = h1[1]

# %%
binsc = bins[0:-1] + np.diff(bins)

# %%
plt.hist(hs, bins='auto', density=True)
plt.plot(binsc, Y)

# %%
plt.hist(hs, bins='auto', density=True)
plt.hist(hs, bins='auto', density=True, cumulative=True, histtype='step')


