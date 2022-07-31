# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:38:52 2022

@author: imargolin
"""

import numpy as np
import torch

def create_ordinal_cost_matrix(size):
    
    cost_matrix = np.ones([size,size])
    for i in range(size):
        for j in range(size):
            cost_matrix[i,j] = np.abs(i-j)
    np.fill_diagonal(cost_matrix, 20)
    return torch.tensor(cost_matrix,dtype=torch.float32)