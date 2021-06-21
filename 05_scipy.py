#!/usr/bin/env python
# coding: utf-8

# # Scipy

# The SciPy framework builds on top of the low-level NumPy framework for multidimensional arrays, and provides a large number of higher-level scientific algorithms. Some of the topics that SciPy covers are:
# 
# * Special functions ([scipy.special](http://docs.scipy.org/doc/scipy/reference/special.html))
# * Integration ([scipy.integrate](http://docs.scipy.org/doc/scipy/reference/integrate.html))
# * Optimization ([scipy.optimize](http://docs.scipy.org/doc/scipy/reference/optimize.html))
# * Interpolation ([scipy.interpolate](http://docs.scipy.org/doc/scipy/reference/interpolate.html))
# * Fourier Transforms ([scipy.fftpack](http://docs.scipy.org/doc/scipy/reference/fftpack.html))
# * Signal Processing ([scipy.signal](http://docs.scipy.org/doc/scipy/reference/signal.html))
# * Linear Algebra ([scipy.linalg](http://docs.scipy.org/doc/scipy/reference/linalg.html))
# * Sparse Eigenvalue Problems ([scipy.sparse](http://docs.scipy.org/doc/scipy/reference/sparse.html))
# * Statistics ([scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html))
# * Multi-dimensional image processing ([scipy.ndimage](http://docs.scipy.org/doc/scipy/reference/ndimage.html))
# * File IO ([scipy.io](http://docs.scipy.org/doc/scipy/reference/io.html))
# 
# Each of these submodules provides a number of functions and classes that can be used to solve problems in their respective topics.
# 
# In this lecture we will look at how to use some of these subpackages.
# 
# To access the SciPy package in a Python program, we start by importing everything from the `scipy` module.

# In[1]:


from scipy import *


# If we only need to use part of the SciPy framework we can selectively include only those modules we are interested in. For example, to include the linear algebra package under the name `la`, we can do:

# In[2]:


import scipy.linalg as la


# ## Fourier transform

# Fourier transforms are one of the universal tools in computational physics, which appear over and over again in different contexts. SciPy provides functions for accessing the classic [FFTPACK](http://www.netlib.org/fftpack/) library from NetLib, which is an efficient and well tested FFT library written in FORTRAN. The SciPy API has a few additional convenience functions, but overall the API is closely related to the original FORTRAN library.
# 
# To use the `fftpack` module in a python program, include it using:

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import *


# To demonstrate how to do a fast Fourier transform with SciPy, let's look at the FFT of the solution to the damped oscillator from the previous section:

# In[4]:


# Seed the random number generator
np.random.seed(1234)

time_step = 0.02
period = 5.

time_vec = np.arange(0, 20, time_step)
sig = (np.sin(2 * np.pi / period * time_vec)
       + 0.5 * np.random.randn(time_vec.size))

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')


# In[5]:


# The FFT of the signal
sig_fft = fft(sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)**2

# The corresponding frequencies
sample_freq = fftfreq(sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection


# In[6]:


high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')


# ## Linear algebra

# The linear algebra module contains a lot of matrix related functions, including linear equation solving, eigenvalue solvers, matrix functions (for example matrix-exponentiation), a number of different decompositions (SVD, LU, cholesky), etc. 
# 
# Detailed documetation is available at: http://docs.scipy.org/doc/scipy/reference/linalg.html

# ### Linear equation systems

# Linear equation systems on the matrix form
# 
# $A x = b$
# 
# where $A$ is a matrix and $x,b$ are vectors can be solved like:

# In[7]:


from scipy.linalg import *


# In[53]:


A = np.array([[1,2,3], [4,5,6], [7,8,9]])
b = np.array([1,2,3])


# In[54]:


x = solve(A, b)

x


# In[55]:


# check
np.dot(A, x) - b


# We can also do the same with
# 
# $A X = B$
# 
# where $A, B, X$ are matrices:

# In[56]:


A = np.random.rand(3,3)
B = np.random.rand(3,3)


# In[12]:


X = solve(A, B)


# In[13]:


X


# ### Eigenvalues and eigenvectors

# The eigenvalue problem for a matrix $A$:
# 
# $\displaystyle A v_n = \lambda_n v_n$
# 
# where $v_n$ is the $n$th eigenvector and $\lambda_n$ is the $n$th eigenvalue.
# 
# To calculate eigenvalues of a matrix, use the `eigvals` and for calculating both eigenvalues and eigenvectors, use the function `eig`:

# In[15]:


evals = eigvals(A)


# In[16]:


evals


# In[17]:


evals, evecs = eig(A)


# In[18]:


evals


# In[19]:


evecs


# The eigenvectors corresponding to the $n$th eigenvalue (stored in `evals[n]`) is the $n$th *column* in `evecs`, i.e., `evecs[:,n]`. To verify this, let's try mutiplying eigenvectors with the matrix and compare to the product of the eigenvector and the eigenvalue:

# In[20]:


n = 1

norm(np.dot(A, evecs[:,n]) - evals[n] * evecs[:,n])


# There are also more specialized eigensolvers, like the `eigh` for Hermitian matrices. 

# ### Matrix operations

# In[21]:


# the matrix inverse
inv(A)


# In[22]:


# determinant
det(A)


# In[23]:


# norms of various orders
norm(A, ord=2), norm(A, ord=Inf)


# ## Optimization

# Optimization (finding minima or maxima of a function) is a large field in mathematics, and optimization of complicated functions or in many variables can be rather involved. Here we will only look at a few very simple cases. For a more detailed introduction to optimization with SciPy see: http://scipy-lectures.github.com/advanced/mathematical_optimization/index.html
# 
# To use the optimization module in scipy first include the `optimize` module:

# In[24]:


from scipy import optimize


# ### Finding a minima

# Let's first look at how to find the minima of a simple function of a single variable:

# In[25]:


def f(x):
    return 4*x**3 + (x-2)**2 + x**4


# In[26]:


fig, ax  = plt.subplots()
x = np.linspace(-5, 3, 100)
ax.plot(x, f(x));


# We can use the `fmin_bfgs` function to find the minima of a function:

# In[27]:


optimize.fmin_bfgs(f, 0.5) 


# We can also use the `brent` or `fminbound` functions. They have a bit different syntax and use different algorithms. 

# In[28]:


optimize.brent(f)


# In[29]:


optimize.fminbound(f, -4, 2)


# ### Finding a solution to a function

# To find the root for a function of the form $f(x) = 0$ we can use the `fsolve` function. It requires an initial guess: 

# In[30]:


omega_c = 3.0
def f(omega):
    # a transcendental equation: resonance frequencies of a low-Q SQUID terminated microwave resonator
    return np.tan(2*pi*omega) - omega_c/omega


# In[31]:


fig, ax  = plt.subplots(figsize=(10,4))
x = np.linspace(0.1, 3, 1000)
y = f(x)
mask = np.where(abs(y) > 50)
x[mask] = y[mask] = NaN # get rid of vertical line when the function flip sign
ax.plot(x, y)
ax.plot([0, 3], [0, 0], 'k')
ax.set_ylim(-5,5);


# In[32]:


optimize.fsolve(f, 0.1)


# In[33]:


optimize.fsolve(f, 0.6)


# In[34]:


optimize.fsolve(f, 1.1)


# ## Interpolation

# Interpolation is simple and convenient in scipy: The `interp1d` function, when given arrays describing X and Y data, returns and object that behaves like a function that can be called for an arbitrary value of x (in the range covered by X), and it returns the corresponding interpolated y value:

# In[35]:


from scipy.interpolate import *


# In[36]:


def f(x):
    return np.sin(x)


# In[37]:


n = np.arange(0.1, 10)  
x = np.linspace(0.1, 9, 100)

y_meas = f(n) + 0.1 * np.random.randn(len(n)) # simulate measurement with noise
y_real = f(x)

linear_interpolation = interp1d(n, y_meas)
y_interp1 = linear_interpolation(x)

cubic_interpolation = interp1d(n, y_meas, kind='cubic')
y_interp2 = cubic_interpolation(x)


# In[38]:


fig, ax = plt.subplots(figsize=(10,4))
ax.plot(n, y_meas, 'bs', label='noisy data')
ax.plot(x, y_real, 'k', lw=2, label='true function')
ax.plot(x, y_interp1, 'r', label='linear interp')
ax.plot(x, y_interp2, 'g', label='cubic interp')
ax.legend(loc=3);


# ## Statistics

# The `scipy.stats` module contains a large number of statistical distributions, statistical functions and tests. For a complete documentation of its features, see http://docs.scipy.org/doc/scipy/reference/stats.html.
# 
# There is also a very powerful python package for statistical modelling called statsmodels. See http://statsmodels.sourceforge.net for more details.

# In[39]:


from scipy import stats


# In[40]:


# create a (discreet) random variable with poissionian distribution

X = stats.poisson(3.5) # photon distribution for a coherent state with n=3.5 photons


# In[41]:


n = np.arange(0,15)

fig, axes = plt.subplots(3,1, sharex=True)

# plot the probability mass function (PMF)
axes[0].step(n, X.pmf(n))

# plot the commulative distribution function (CDF)
axes[1].step(n, X.cdf(n))

# plot histogram of 1000 random realizations of the stochastic variable X
axes[2].hist(X.rvs(size=1000));


# In[42]:


# create a (continous) random variable with normal distribution
Y = stats.norm()


# In[43]:


x = np.linspace(-5,5,100)

fig, axes = plt.subplots(3,1, sharex=True)

# plot the probability distribution function (PDF)
axes[0].plot(x, Y.pdf(x))

# plot the commulative distributin function (CDF)
axes[1].plot(x, Y.cdf(x));

# plot histogram of 1000 random realizations of the stochastic variable Y
axes[2].hist(Y.rvs(size=1000), bins=50);


# Statistics:

# In[44]:


X.mean(), X.std(), X.var() # poission distribution


# In[45]:


Y.mean(), Y.std(), Y.var() # normal distribution


# ### Statistical tests

# Test if two sets of (independent) random data comes from the same distribution:

# In[46]:


t_statistic, p_value = stats.ttest_ind(X.rvs(size=1000), X.rvs(size=1000))

print("t-statistic =", t_statistic)
print("p-value =", p_value)


# Since the p value is very large we cannot reject the hypothesis that the two sets of random data have *different* means.

# To test if the mean of a single sample of data has mean 0.1 (the true mean is 0.0):

# In[47]:


stats.ttest_1samp(Y.rvs(size=1000), 0.1)


# Low p-value means that we can reject the hypothesis that the mean of Y is 0.1.

# In[48]:


Y.mean()


# In[49]:


stats.ttest_1samp(Y.rvs(size=1000), Y.mean())


# ## Statistics

# The `scipy.signal` module contains a large number of signal analysis functions. For a complete documentation of its features, see https://docs.scipy.org/doc/scipy/reference/signal.html.

# In[50]:


# scipy.signal.resample(): resample a signal to n points using FFT.
t = np.linspace(0, 5, 100)
x = np.sin(t)

from scipy import signal
x_resampled = signal.resample(x, 25)

plt.plot(t, x) 

plt.plot(t[::4], x_resampled, 'ko')


# In[51]:


# scipy.signal.detrend(): remove linear trend from signal:
t = np.linspace(0, 5, 100)
x = t + np.random.normal(size=100)

from scipy import signal
x_detrended = signal.detrend(x)

plt.plot(t, x) 

plt.plot(t, x_detrended) 


# In[52]:


# Cross correlated signal
from scipy import signal
import matplotlib.pyplot as plt
rng = np.random.default_rng()

sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
sig_noise = sig + rng.standard_normal(len(sig))
corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128

clock = np.arange(64, len(sig), 128)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.plot(clock, sig[clock], 'ro')
ax_orig.set_title('Original signal')
ax_noise.plot(sig_noise)
ax_noise.set_title('Signal with noise')
ax_corr.plot(corr)
ax_corr.plot(clock, corr[clock], 'ro')
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)
fig.tight_layout()

