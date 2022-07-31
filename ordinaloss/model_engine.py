# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:38:37 2022

@author: imargolin
"""

from torch.optim import Adam
import torch.nn.functional as F
import torch
from tqdm import tqdm



class OrdinalEngine:
    def __init__(self, model, loss_fn, device, optimizer_fn = Adam, **optimizer_params):
        self.device = device
        
        self.model = model
        self.model.to(self.device)
        
        self.optimizer = self.set_optimizer(optimizer_fn, **optimizer_params)
        self.loss_fn = loss_fn #Should get y_pred and y_true, becomes a method.
        self.loss_fn.to(self.device)
        
    def set_optimizer(self, optimizer_fn, **optimizer_params):
        self._optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        
        
    def _train_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        self.model.train()
        self._optimizer.zero_grad()
        stats = self.forward(X, y)
        
        stats["loss"].backward()
        self._optimizer.step()
        return stats
        
    
    def forward(self, X, y):
        #Forward and extract data
        y_pred = F.softmax(self.model(X), dim = 1) #Normalized
        loss = self.loss_fn(y_pred, y)
        accuracy = (y_pred.argmax(axis=1) == y).to(torch.float32).mean()
        return {"loss": loss, "accuracy": accuracy, "batch_size": X.shape[0]} 
        
    def _train_epoch(self, loader):
        iterator = tqdm(loader, total = len(loader))
        
        cum_batch_size = 0 
        cum_loss = 0
        cum_accuracy = 0 
        
        
        for X, y in iterator:
            stats = self._train_batch(X, y)
            cum_batch_size += stats["batch_size"]
            cum_loss += stats["loss"].item() * stats["batch_size"]
            cum_accuracy += stats["accuracy"].item() * stats["batch_size"] #Total corrected so far
            
            iterator.set_postfix(loss= cum_loss / cum_batch_size, accuracy = cum_accuracy / cum_batch_size, batch_size = cum_batch_size)
            
            
    def _eval_batch(self, X, y):
        self.model.eval()
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            stats = self.forward(X, y)

        return stats
            
    def _eval_epoch(self, loader):
        
        cum_batch_size = 0 
        cum_loss = 0
        cum_accuracy = 0 
        
        for X, y in loader:
            stats = self._eval_batch(X, y)
            cum_batch_size += stats["batch_size"]
            cum_loss += stats["loss"].item() * stats["batch_size"]
            cum_accuracy += stats["accuracy"].item() * stats["batch_size"] #Total corrected so far
            
        return {
                "loss": cum_loss/cum_batch_size, 
                "accuracy":cum_accuracy/cum_batch_size
               }
        
    def train(self, train_loader, test_loader=None, n_epochs=1):
        for _ in range(n_epochs):
            self._train_epoch(train_loader)
            if test_loader:
                print(self._eval_epoch(test_loader))
