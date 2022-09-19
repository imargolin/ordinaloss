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
import mlflow
from  mlflow.tracking import MlflowClient

print(f"loaded {__name__}")


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
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print ('LR is set to {}'.format(lr))
        
        return optimizer

from torch.nn import L1Loss

class RegressionEngine:
    def __init__(self, model, device, optimizer_fn=Adam, **optimizer_params):
        self.device = device
        self.model = model
        self.model.to(self.device)

        self.loss_fn = L1Loss()

        self.set_optimizer(optimizer_fn, **optimizer_params)
        self.epoch_trained = 0

    def set_optimizer(self, optimizer_fn, **optimizer_params):
        self._optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)

    def forward(self, X, y):
        #Forward and extract data
        y_pred = self.model(X)[:, 0]
        loss = self.loss_fn(y_pred, y)
        
        #numpying
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        y_pred = y_pred.clip(-0.499, 5.4999).round()
        accuracy = (y_pred == y).mean()

        stats = {"loss": loss, 
                "accuracy": accuracy, 
                "batch_size": X.shape[0]} 
        
        return stats

    def _train_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        self.model.train()
        self._optimizer.zero_grad()
        
        stats = self.forward(X, y)
        stats["loss"].backward()

        self._optimizer.step()
        return stats

        



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

        self.init_mlrun()
        mlflow.log_params(optimizer_params)
        mlflow.log_param("optimizer_fn", optimizer_fn.__name__)
        mlflow.log_param("lr_scheduler", self.use_lr_scheduler)
        mlflow.log_param("loss_fn", loss_fn.__repr__())
        mlflow.log_param("model_name", self.model._get_name())
        mlflow.log_param("n_layers", len(list(self.model.parameters())))
        
        #This is a crucial parameter
        if hasattr(self.loss_fn, "cb_matrix"):
            mlflow.log_param("cb_matrix", loss_fn.cb_matrix)
    
    def init_mlrun(self):

        mlflow.end_run() #Ending run if exists.

        experiment_names = [experiment.name for experiment in mlflow.list_experiments()]
        if "OrdinalLoss" not in experiment_names:
            mlflow.create_experiment("OrdinalLoss")

        experiment_id = mlflow.get_experiment_by_name("OrdinalLoss").experiment_id
        mlflow.start_run(experiment_id=experiment_id)
        
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
        iterator = tqdm(loader, total = len(loader), desc= f"Training, epoch {self.epochs_trained}")
        
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
            cum_mae += stats["mae"].item() *n

            epoch_loss = cum_loss / cum_batch_size
            epoch_accuracy = cum_accuracy / cum_batch_size
            epoch_mae = cum_mae / cum_batch_size
            epcoh_batch_size = cum_batch_size

            iterator.set_postfix(loss= epoch_loss, 
                                 accuracy = epoch_accuracy, 
                                 mae = epoch_mae,
                                 batch_size = epcoh_batch_size)

        self.epochs_trained +=1
        
        if self.use_lr_scheduler:
            self.scheduler.step()

        metrics = {"mae":epoch_mae, "loss":epoch_loss, "accuracy":epoch_accuracy}
        self.log_metrics(metrics, phase = "train")

           
    def _eval_batch(self, X, y):
        '''
        Evaluating the batch
        '''

        self.model.eval()
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            stats = self.forward(X, y)

        return stats
            
    def _eval_epoch(self, loader, log_metrics = False):
        '''
        Evaluating the entire epoch,
        logging the metrics only if under self.train() scope.
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

            epoch_loss = cum_loss / cum_batch_size
            epoch_accuracy = cum_accuracy / cum_batch_size
            epoch_mae = cum_mae / cum_batch_size
            epcoh_batch_size = cum_batch_size
            
            iterator.set_postfix(loss= epoch_loss, 
                                 accuracy = epoch_accuracy, 
                                 mae = epoch_mae,
                                 batch_size = epcoh_batch_size)
            
        metrics = {"mae":epoch_mae, "loss":epoch_loss, "accuracy":epoch_accuracy}

        if log_metrics:
            self.log_metrics(metrics, phase = "test")        
            
        return metrics

    def log_metrics(self, metrics, phase):
        '''
        metrics is a dictionary with accuracy, mae and loss
        '''
        metric_to_log = {f"{k}_{phase}": v for k,v in metrics.items()}
        mlflow.log_metrics(metric_to_log, step = self.epochs_trained)
        
    def train(self, train_loader, test_loader=None, n_epochs=1):
        for _ in range(n_epochs):
            self._train_epoch(train_loader)
            if test_loader:
                print(self._eval_epoch(test_loader, log_metrics=True))
