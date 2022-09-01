# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:38:37 2022

@author: imargolin
"""

from torch.optim import Adam
import torch.nn.functional as F
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler


class LRScheduler():
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10, 
                 lr_decay_factor = 0.9):

        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor every lr_decay_epoch epochs.'''
        lr = self.init_lr * (self.lr_decay_factor ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print ('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print ('LR is set to {}'.format(lr))
        
        return optimizer


class OrdinalEngine:
    def __init__(self, model, loss_fn, device, optimizer_fn = Adam, 
                 use_lr_scheduler = False, scheduler_lambda_fn = None , **optimizer_params):

        self.device = device
        
        self.model = model
        self.model.to(self.device)
        
        self.set_optimizer(optimizer_fn, **optimizer_params)
        self.use_lr_scheduler = use_lr_scheduler
        
        if self.use_lr_scheduler:
            self.scheduler = lr_scheduler.LambdaLR(self._optimizer, scheduler_lambda_fn) 

        self.loss_fn = loss_fn #Should get y_pred and y_true, becomes a method.
        self.loss_fn.to(self.device)
        self.epochs_trained = 0
        
    def set_optimizer(self, optimizer_fn, **optimizer_params):
        '''
        Setting an optimizer, happens in the __init__
        '''
        
        self._optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        
    def forward(self, X, y):
        #Forward and extract data
        y_pred = F.softmax(self.model(X), dim = 1) #Normalized
        loss = self.loss_fn(y_pred, y)
        y_arg_pred = y_pred.argmax(axis=1)
        
        accuracy = (y_arg_pred == y).to(torch.float32).mean()
        mae = (y_arg_pred- y).to(torch.float32).abs().mean()
        
        
        return {"loss": loss, 
                "accuracy": accuracy, 
                "mae": mae,
                "batch_size": X.shape[0]} 
        
    def _train_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        self.model.train()
        self._optimizer.zero_grad()
        stats = self.forward(X, y)
        
        stats["loss"].backward()
        self._optimizer.step()
        return stats

    def _train_epoch(self, loader):
        iterator = tqdm(loader, total = len(loader))
        
        cum_batch_size = 0 
        cum_loss = 0
        cum_accuracy = 0 
        cum_mae = 0

        for X, y in iterator: 
            stats = self._train_batch(X, y)
            n = stats["batch_size"]
            
            cum_batch_size += n
            cum_loss += stats["loss"].item() * n
            cum_accuracy += stats["accuracy"].item() * n #Total corrected so far
            cum_mae += stats["mae"].item()  *n
            iterator.set_postfix(loss= cum_loss / cum_batch_size, 
                                 accuracy = cum_accuracy / cum_batch_size, 
                                 mae = cum_mae / cum_batch_size,
                                 batch_size = cum_batch_size)

        self.epochs_trained +=1
        
        if self.use_lr_scheduler:
            self.scheduler.step()
           
    def _eval_batch(self, X, y):
        '''
        Evaluating the batch
        '''

        self.model.eval()
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            stats = self.forward(X, y)

        return stats
            
    def _eval_epoch(self, loader):
        '''
        Evaluating the entire epoch
        '''

        iterator = tqdm(loader, total = len(loader))
        
        cum_batch_size = 0 
        cum_loss = 0
        cum_accuracy = 0 
        cum_mae = 0
        
        for X, y in iterator:
            stats = self._eval_batch(X, y)
            n = stats["batch_size"]
            
            cum_batch_size += n
            cum_loss += stats["loss"].item() * n
            cum_accuracy += stats["accuracy"].item() * n #Total corrected so far
            cum_mae += stats["mae"].item()  *n
            iterator.set_postfix(loss= cum_loss / cum_batch_size, 
                                 accuracy = cum_accuracy / cum_batch_size, 
                                 mae = cum_mae / cum_batch_size,
                                 batch_size = cum_batch_size)
            
        return {
                "loss": cum_loss/cum_batch_size, 
                "accuracy":cum_accuracy/cum_batch_size,
                "mae":cum_mae / cum_batch_size,
                
               }
        
    def train(self, train_loader, test_loader=None, n_epochs=1):
        for _ in range(n_epochs):
            self._train_epoch(train_loader)
            if test_loader:
                print(self._eval_epoch(test_loader))
