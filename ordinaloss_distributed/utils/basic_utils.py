# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:38:52 2022

@author: imargolin
"""

import numpy as np
import torch

from torch import nn
from torchvision import models

def satisfy_constraints(test_dist, constraints):
    return (test_dist < constraints).all().item()

# def modify_loss_function(constraints, test_dist, current_lambdas):
#     #constraints -> [0.2, 1, 1, 1, 1]
#     #test_dist ->   [0.3, 0, 0, 0, 0.7]

#     c = len(test_dist)
#     ratios = test_dist / constraints
#     violations = (ratios>1).to(torch.float32)
#     ratios = [ratios[i] for i in range(c) if violations[i]==0 else 1]
#     ratios = torch.tensor(ratios, dtype=torch.float32)
    
#     new_lambdas = current_lambdas *ratios
#     new_lambdas = new_lambdas/new_lambdas.sum()


#     return new_lambdas

def modify_lambdas(constraints, test_dist, current_lambdas, meta_learning_rate = 0.1):
    #constraints -> [0.2, 1.0, 1.0, 1.0, 1.0]
    #test_dist ->   [0.3, 0.0, 0.0, 0.0, 0.7]

    diffs = test_dist - constraints          #[0.1, -1.0, -1.0, -1.0, -0.3]
    violations = (diffs>0).to(torch.float32) #[1.0,  0.0,  0.0,  0.0,  0.0]
    step = diffs*violations                  #[0.1,  0.0,  0.0,  0.0,  0.0]
    
    new_lambdas = current_lambdas + step*meta_learning_rate
    #new_lambdas = new_lambdas/new_lambdas.sum()

    return new_lambdas    