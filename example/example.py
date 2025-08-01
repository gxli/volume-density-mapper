#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from astropy.io import fits
from astropy import constants as cons
from volume_density_mapper import *


# In[37]:


nh = fits.getdata('IC348_nh.fits')
header = fits.getheader('IC348_nh.fits')
mh2 = 1.34*3.34e-24
pc = 3.08e18

plt.figure(dpi = 100)
plt.imshow(np.log10(nh * mh2), origin = 'lower')
plt.colorbar(label=r'Log(surface density ($\rm g cm^{-2}$))')


# In[45]:


# charactersitic scale (width) measurements
input_map = nh.copy() * mh2
dx = header['CDELT2']/180*np.pi*270 * pc
#pixel size, the same unit with that of output
density, width = compute_mean_density_width(input_map, dx)

plt.figure(dpi = 100)
plt.imshow(np.log10(density), origin = 'lower')
plt.colorbar(label = r'log(Volume Density (r$g\;cm^{-3}$))')


plt.figure(dpi = 100)
plt.imshow(np.log10(width), origin = 'lower',cmap = 'magma')
plt.colorbar(label = r'log(width (cm))')

# plt.contour(np.log10(nh), np.arange(21.6,22.6,0.3), colors = 'k')


# In[46]:


# restructure the density structure in 3D space

data_in = nh * mh2 # convert to cgs unit 
dx = header['CDELT2']/180*np.pi*270 * pc #pixel size, unit as cm (cgs unit)
data3d = density_reconstruction_3d(data_in, dx)


print(np.shape(data3d))


plt.show()
# In[55]:



# In[ ]:




