# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 14:43:52 2022

@author: imargolin
"""

import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np

def create_ordinal_cost_matrix(size):
    
    cost_matrix = np.ones([size,size])
    for i in range(size):
        for j in range(size):
            cost_matrix[i,j] = np.abs(i-j)
    np.fill_diagonal(cost_matrix, 20)
    return torch.tensor(cost_matrix,dtype=torch.float32)


class CSCELoss(nn.Module):
    def __init__(self, 
                 cb_matrix: torch.tensor, 
                 smoothing=1e-7):
        super().__init__()
        self.cb_matrix = cb_matrix
        self.n_classes = cb_matrix.shape[0]
        self.smoothing = smoothing
        
    def forward(self,y_pred, y_true):
        
        #y_pred is going under smoothing,
        y_pred = y_pred * (1-self.smoothing) + (1/self.n_classes) * self.smoothing
        
        #y_pred shape is [batch_size, num_classes]
        #y_true indexes of the [batch_size, ] 
        weights = self.cb_matrix[y_true]
        
        y_true = F.one_hot(y_true, num_classes = self.n_classes)
        
        
        loss = (y_true * torch.log(y_pred)) + ((1-y_true) * torch.log(1-y_pred))
        
        return -1* (loss * weights).sum(axis=1).mean() #Returns as avg across all samples
    
    def to(self, device):
        self.cb_matrix = self.cb_matrix.to(device)