# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:38:52 2022

@author: imargolin
"""

import numpy as np
import torch
import numbers
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def create_mock_model(num_classes=5, num_features=2000):
    return nn.Linear(num_features, num_classes)

def create_mock_dsets(
    num_classes=5, 
    num_features=2000, 
    num_samples=500, 
    phases=["train", "test", "val", "auto_test"]
    ):

    out={}
    for phase in phases:
        dset = TensorDataset(torch.rand(num_samples, num_features), torch.randint(num_classes, size=(num_samples,)))
        out[phase] = dset

    return out


def get_only_metrics(my_dict:dict):
    return {k:v for k,v in my_dict.items() if isinstance(v, numbers.Number)}

def satisfy_constraints(test_dist:torch.Tensor, constraints: torch.Tensor) -> bool:
    """Returns an answer to the question whehter constraints has been satisfied.

    Args:
        test_dist (torch.Tensor): The prediction destribution, should be summed to 1.
        constraints (torch.Tensor): The constraint for each class. should satisfy from below. 

    Returns:
        bool: True/False if all constraints are satisfied.
    """

    return (test_dist <= constraints).all().item()

@torch.no_grad()
def modify_lambdas(
    constraints: torch.Tensor, 
    test_dist: torch.Tensor, 
    current_lambdas: torch.Tensor, 
    meta_learning_rate: float = 0.1
    ) -> torch.Tensor: 
    """_summary_

    Args:
        constraints (torch.Tensor): _description_
        test_dist (torch.Tensor): _description_
        current_lambdas (torch.Tensor): _description_
        meta_learning_rate (float, optional): How to update the current lambdas based on the diffs. Defaults to 0.1.

    Returns:
        torch.Tensor: _description_
    """    ''''''
    

    #constraints -> [0.2, 1.0, 1.0, 1.0, 1.0]
    #test_dist ->   [0.3, 0.0, 0.0, 0.0, 0.7]

    diffs = test_dist - constraints          #[0.1, -1.0, -1.0, -1.0, -0.3]
    violations = (diffs>0).to(torch.float32) #[1.0,  0.0,  0.0,  0.0,  0.0]
    step = diffs*violations                  #[0.1,  0.0,  0.0,  0.0,  0.0]
    
    #Changing the current lambdas (return a new tensor)
    new_lambdas = current_lambdas + step*meta_learning_rate 

    return new_lambdas.to(torch.float32)

@torch.no_grad()
def modify_lambdas_extended(
    constraints: torch.Tensor, 
    test_dist: torch.Tensor, 
    current_lambdas: torch.Tensor, 
    meta_learning_rate: float = 0.1,
    budget_booster: float = 10.0
    ) -> torch.Tensor: 
    
    current_budget = current_lambdas.sum() #This is the current budget. 
    new_budget = current_budget + budget_booster
    step = torch.nn.functional.relu(test_dist - constraints)
    new_lambdas = current_lambdas + step*meta_learning_rate
    new_lambdas = new_lambdas/new_lambdas.sum() #Should be normalized now
    new_lambdas = new_lambdas*new_budget

    return new_lambdas.to(torch.float32)
