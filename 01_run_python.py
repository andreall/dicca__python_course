#!/usr/bin/env python
# coding: utf-8

# # Running Python code

# ## Interactively (demo)

# To run Python code interactively, one can use the standard Python prompt, which can be launched by typing ``python`` in your standard shell:
# 
#     $ python
#     Python 3.4.1 (default, May 21 2014, 21:17:51) 
#     [GCC 4.2.1 Compatible Apple Clang 4.1 ((tags/Apple/clang-421.11.66))] on darwin
#     Type "help", "copyright", "credits" or "license" for more information.
#     >>>
# 
# The ``>>>`` indicates that Python is ready to accept commands. If you type ``a = 1`` then press enter, this will assign the value ``1`` to ``a``. If you then type ``a`` you will see the value of ``a`` (this is equivalent to ``print a``):
# 
#     >>> a = 1
#     >>> a
#     1
# 
# The Python shell can execute any Python code, even multi-line statements, though it is often more convenient to use Python non-interactively for such cases.
# 
# The default Python shell is limited, and in practice, you will want instead to use the IPython (or interactive Python) shell. This is an add-on package that adds many features to the default Python shell, including the ability to edit and navigate the history of previous commands, as well as the ability to tab-complete variable and function names. To start up IPython, type:
# 
#     $ ipython
#     Python 3.4.1 (default, May 21 2014, 21:17:51) 
#     Type "copyright", "credits" or "license" for more information.
# 
#     IPython 2.1.0 -- An enhanced Interactive Python.
#     ?         -> Introduction and overview of IPython's features.
#     %quickref -> Quick reference.
#     help      -> Python's own help system.
#     object?   -> Details about 'object', use 'object??' for extra details.
# 
#     In [1]:
# 
# The first time you start up IPython, it will display a message which you can skip over by pressing ``ENTER``. The ``>>>`` symbols are now replaced by ``In [x]``, and output, when present, is prepended with ``Out [x]``. If we now type the same commands as before, we get:
# 
#     In [1]: a = 1
# 
#     In [2]: a
#     Out[2]: 1
# 
# If you now type the up arrow twice, you will get back to ``a = 1``.

# ## Running scripts (demo)

# While the interactive Python mode is very useful to exploring and trying out code, you will eventually want to write a script to record and reproduce what you did, or to do things that are too complex to type in interactively (defining functions, classes, etc.). To write a Python script, just use your favorite code editor to put the code in a file with a ``.py`` extension. For example, we can create a file called ``test.py`` containing:
# 
#     a = 1
#     print(a)
# 
# On Linux computers you can use for example the ``emacs`` editor which you can open by typing:
#     
#     emacs &
#     
# (ignore the warnings that it prints to the terminal).
# 
# We can then run the script on the command-line with:
# 
#     $ python test.py
#     1
# 
# Note: The ``print`` statement is necessary, because typing ``a`` on its own will only print out the value in interactive mode. In scripts, the printing has to be explicitly requested with the print command. To print multiple variables, just separate them with a comma after the print command:
# 
#     print(a, 1.5, "spam")

# ## Combining interactive and non-interactive use (demo)

# It can sometimes be useful to run a script to set things up, and to continue in interactive mode. This can be done using the ``%run`` IPython command to run the script, which then gets executed. The IPython session then has access to the last state of the variables from the script:
# 
#     $ ipython
#     Python 3.4.1 (default, May 21 2014, 21:17:51) 
#     Type "copyright", "credits" or "license" for more information.
# 
#     IPython 2.1.0 -- An enhanced Interactive Python.
#     ?         -> Introduction and overview of IPython's features.
#     %quickref -> Quick reference.
#     help      -> Python's own help system.
#     object?   -> Details about 'object', use 'object??' for extra details.
# 
#     In [1]: %run test.py
#     1
# 
#     In [2]: a + 1
#     Out[2]: 2

# # Using the IPython notebook

# The IPython *notebook* is allows you to write notebooks similar to e.g. Mathematica. The advantage of doing this is that you can include text, code, and plots in the same document. This makes it ideal for example to write up a report about a project that uses mostly Python code, in order to share with others. In fact, the notes for this course are written using the IPython notebook!

# Click on ``New Notebook`` on the right, which will start a new document. You can change the name of the document by clicking on the **Untitled** name at the top and entering a new name. Make sure you then save the document (make sure that you save regularly as you might lose content if you close the browser window!).
# 
# At first glance, a notebook looks like a fairly typical application - it has a menubar (File, Edit, View, etc.) and a tool bar with icons. Below this, you will see an empty cell, in which you can type any Python code. You can write several lines of code, and once it is ready to run, you can press shift-enter and it will get executed:

# In[ ]:


a = 1
print(a)


# You can then click on that cell, change the Python code, and press shift-enter again to re-execute the code. Once you have executed a cell once, a new cell will appear below. You can again enter some code, then press shift-enter to execute it.

# ## Text

# It is likely that you will want to enter actual text (non-code) in the notebook. To do this, click on a cell, and in the drop-down menu in the toolbar, select 'Markdown'. This is a specific type of syntax for writing text. You can just write text normally and press shift-enter to *render* it:
# 
#     This is some plain text
# 
# To edit it, double click on the cell. You can also enter section headings using the following syntax:
# 
#     This is a title
#     ===============
# 
#     This is a sub-title
#     -------------------
# 
# which will look like:
# 
# This is a title
# ===============
# 
# This is a sub-title
# -------------------
# 
# Finally, if you are familiar with LaTeX, you can enter equations using:
# 
#     $$E = m c^2$$
# 
# on a separate line, or:
# 
#     The equation $p=h/\lambda$ is very important
# 
# to include it in a sentence. This will look like:
# 
# $$E = m c^2$$
# 
# The equation $p=h/\lambda$ is very important
# 
# For more information about using LaTeX for equations, see [this guide](http://en.wikibooks.org/wiki/LaTeX/Mathematics).

# ## Important notes

# A few important notes about using the notebook:
# 
# * Save often! There is an auto-save in the notebook, but better to also save explicitly from time to time.
# 
# * Code *can* be executed in an order different from top to bottom, but note that if you do this variables will not be reset. So for example if you type:

# In[1]:


a = 1


# then go higher up and type:

# In[2]:


print(a)


# it will give the value you previously set. To make sure that your code works from top to bottom, go to the 'Cell' menu item and go to **All Output** -> **Clear** then in the **Cell** menu, select **Run All**.
# 
# In addition, even if you remove a cell, then variables set in that cell still exist unless you restart the notebook. If you want to restart a notebook, you can select **Kernel** -> **Restart**. This removes any variables from memory, and you have to start running the notebook from the start.
