#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import e, sqrt, pi


# In[2]:


try:
    import matplotlib

    #matplotlib.use('agg')  # no interactive plotting, only save figures
    #import pylab
    from matplotlib import pyplot as plt
    ## This import registers the 3D projection, but is otherwise unused.
    #from mpl_toolkits.mplot3d import Axes3D
    #from mpl_toolkits.mplot3d.art3d import Line3DCollection # noqa: F401 unused import   
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection # noqa: F401 unused import    
    have_matplotlib = True
except ImportError:
    have_matplotlib = False

import csv
import pandas as pd
import scipy.io


# In[5]:


file = 'features.csv'
features_space = pd.read_csv(file)
features_space.shape


# In[18]:


natom = features_space.shape[0]
ndim = features_space.shape[1]

outfile = 'network.csv'
with open(outfile, 'w') as output:
    writer = csv.writer(output, delimiter=' ')    
    for i in list(range(natom)):
        features_i = features_space.iloc[i, :].values
        #print(i,int(features_i[0]),features_i[1])
        for j in list(range(i+1,natom)):
            features_j = features_space.iloc[j, :].values
            #print(i,j,int(features_i[0]), int(features_j[0]))
            dist=0.0
            for k in list(range(1,ndim)):
                dist += (features_i[k] - features_j[k])**2
            print( int(features_i[0]), int(features_j[0]), dist)
            writer.writerow( [int(features_i[0]), int(features_j[0]), dist] )


# In[ ]:




