# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 14:43:52 2022

@author: imargolin
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

print(f"loaded {__name__}")
#device = "cuda:2" if torch.cuda.is_available() else "cpu"

def create_ordinal_cost_matrix(num_classes, cost_distance, 
                               diagonal_value, normalize = True):
    #TODO: Should be have a better specification.
    
    cost_matrix = np.ones([num_classes,num_classes])
    for i in range(num_classes):
        for j in range(num_classes):
            cost_matrix[i,j] = np.abs(i-j) * cost_distance +1
    np.fill_diagonal(cost_matrix, diagonal_value)
    
    if normalize:
        cost_matrix = cost_matrix/cost_matrix.sum(axis=1)[:, None]

    return torch.tensor(cost_matrix,dtype=torch.float32)

class CSCELoss(nn.Module):
    def __init__(self, cb_matrix: torch.tensor, smoothing=1e-7):
        super().__init__()
        self.cb_matrix = cb_matrix
        self.n_classes = cb_matrix.shape[0]
        self.smoothing = smoothing
        
    def forward(self,y_pred, y_true):
        '''
        y_pred is shape of (N, C) - Normalized
        y_true is shape of (N) Integers.
        '''
        
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

class SinimLoss(nn.Module):
    def __init__(self, ordinal_matrix):
        super().__init__()
        self.ordinal_matrix = ordinal_matrix
        self.ordinal_matrix+=1

        self.n_classes = ordinal_matrix.shape[0]
        #Filling diagonal
        for i in range(self.n_classes):
            self.ordinal_matrix[i,i] = 0

        self.ordinal_matrix = self.ordinal_matrix

    def forward(self, y_pred, y_true):
        '''
        y_pred is shape of (N, C) - Normalized
        y_true is shape of (N) Integers.
        '''
        weights = self.ordinal_matrix[y_true]
        y_true = F.one_hot(y_true, num_classes = self.n_classes)
        return torch.sum((y_pred * weights) ** 2) / y_pred.shape[0]

    def to(self, device):
        self.ordinal_matrix = self.ordinal_matrix.to(device)
        return self

class GirlsLoss(nn.Module):
    def __init__(self, ordinal_matrix):
        super().__init__()
        self.ordinal_matrix = ordinal_matrix
        self.ordinal_matrix+=1

        self.n_classes = ordinal_matrix.shape[0]
        #Filling diagonal
        for i in range(self.n_classes):
            self.ordinal_matrix[i,i] = 0

        self.ordinal_matrix = self.ordinal_matrix.cuda()
    
    def forward(self, y_pred, y_true):
        '''

        y_pred is shape of (N, C) - Normalized
        y_true is shape of (N) Integers.
        '''

        weights = self.ordinal_matrix[y_true]
        y_true = F.one_hot(y_true, num_classes = self.n_classes)

        loss = -1 * torch.sum((
                y_true * torch.log(y_pred) + (1 - y_true) *
                torch.log(1 - y_pred))* weights) / y_pred.shape[0]
        return loss

    def to(self, device):
        self.ordinal_matrix = self.ordinal_matrix.to(device)
        return self

class SinimLossOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1

    def forward(self, outputs, labels):
        """
        The outputs should be softmax-normalized!
        """
        #softmax_op = torch.nn.Softmax(1)
        prob_pred = outputs

        cls_weights = np.array([[1, 3, 5, 7, 9],
                                [3, 1, 3, 5, 7],
                                [5, 3, 1, 3, 5],
                                [7, 5, 3, 1, 3],
                                [9, 7, 5, 3, 1]], dtype=np.float)

        cls_weights = cls_weights + 1.0
        np.fill_diagonal(cls_weights, 0)

        batch_num, class_num = outputs.size()

        class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
        labels_np = labels.data.cpu().numpy()
        print(labels_np)
        for ind in range(batch_num):
            print(ind)
            class_hot[ind, :] = cls_weights[labels_np[ind], :]
        class_hot = torch.from_numpy(class_hot)
        class_hot = torch.autograd.Variable(class_hot)
        loss = torch.sum((prob_pred * class_hot) ** 2) / batch_num

        return loss

class GirlsLossOld(nn.Module):
    def __init__(self, ordinal_matrix):
        super().__init__()
        self.ordinal_matrix = ordinal_matrix
    
    def forward(self, outputs, labels):
        '''
        outputs are normalized
        '''
        prob_pred = outputs

        cls_weights = np.array([[1, 3, 5, 7, 9],
                                [3, 1, 3, 5, 7],
                                [5, 3, 1, 3, 5],
                                [7, 5, 3, 1, 3],
                                [9, 7, 5, 3, 1]], dtype=np.float)
        cls_weights = cls_weights + 1.0
        np.fill_diagonal(cls_weights, 0)

        batch_num, class_num = outputs.size()

        class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
        labels_np = labels.data.cpu().numpy()

        y_labels = np.zeros((labels_np.size, 5))
        y_labels[np.arange(labels_np.size), labels_np] = 1
        y_labels = torch.from_numpy(y_labels).cuda()


        for ind in range(batch_num):
            class_hot[ind, :] = cls_weights[labels_np[ind], :]
        class_hot = torch.from_numpy(class_hot)
        class_hot = torch.autograd.Variable(class_hot).cuda()
        #class_hot = torch.ones(size=class_hot.shape)#.to('cuda:0')

        loss = -1 * torch.sum((
                y_labels * torch.log(prob_pred) + (1 - y_labels) *
                torch.log(1 - prob_pred))* class_hot) / batch_num

        return loss.to(torch.float32)

class PredictionLoss(nn.Module):
    def __init__(self, lambdas: torch.tensor):
        super().__init__()
        self.lambdas = lambdas

    def forward(self, y_pred: torch.tensor, y_true=None):
        return self.lambdas.matmul(y_pred.T).mean()

    def to(self, device):
        self.lambdas.to(device)

class CombinedLoss(nn.Module):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
    
    def forward(self, y_pred, y_true):
        return self.l1(y_pred, y_true) + self.l2(y_pred, y_true)

    def to(self, device):
        self.l1.to(device)
        self.l2.to(device)