# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:28:19 2017

@author: Shane
"""

import numpy as np

x = [1, 1, 0, -1, -1, -1, 0, 1]
y = [0, 1, 1, 1, 0, -1, -1, -1]

theta = np.arctan2(y,x)  
print theta	
