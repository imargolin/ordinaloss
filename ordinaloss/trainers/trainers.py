# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:38:37 2022

@author: imargolin
"""

from torch.optim import Adam

from typing import Any, List, Set, Dict, Tuple

import torch.nn.functional as F
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler
import mlflow
from  mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from ordinaloss.utils.metric_utils import RunningMetric, BinCounter, StatsCollector
from ordinaloss.utils.metric_utils import accuracy_pytorch, mae_pytorch, calc_cost_metric
from ordinaloss.utils.basic_utils import get_only_metrics
from sklearn.metrics import accuracy_score, mean_absolute_error
import os
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
import secrets

print(f"loaded {__name__}")

class EarlyStopper:
    def __init__(self, patience:int = 5, min_delta:float =1.0):
        """My small implementation for early stopping.

        Args:
            patience (int, optional): How long to wait after last time validation loss improved. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 1.0
        """
        self.patience = patience
        self.min_delta = min_delta #We want the loss to be 0.99 from the previous epoch.
        self.counter = 0
        self.min_validation_loss = np.inf
        self.is_best_model = False
        self.early_stop = False

    def step(self, loss:float):
        """Stepping the EarlyStopper with one more loss, checks whether should stop.

        Args:
            loss (float): The loss to be monitored
        """

        if loss < self.min_validation_loss * self.min_delta:
            #New loss was found!
            self.counter = 0 #Reset the counter
            self.min_validation_loss = loss
            self.is_best_model = True
        
        else: #Not enough imporvement
            
            self.counter+=1
            print(f"Strike {self.counter} / {self.patience}")
            self.is_best_model = loss < self.min_validation_loss #yet might be new best.

            if self.counter>=self.patience:
                self.early_stop = True

class LRScheduler:
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10, 
                 lr_decay_factor = 0.9):

        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor

    def step(self):
        pass

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

class SingleGPUTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        loaders: Dict[str, DataLoader],
        optimizer: torch.optim.Optimizer, 
        gpu_id: int,
        save_every: int,
        num_classes:int
        ):

        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.loaders = loaders

        self.optimizer = optimizer
        self.save_every = save_every
        self.num_classes = num_classes
        self.epochs_trained = 0
        self.model_id = secrets.token_hex(nbytes=16)
        self.checkpoint_path = f"{self.model_id}.pt"

    def forward(self, X):
        return F.softmax(self.model(X), dim = 1) #Normalized        

    def prepare_input(self, X, y):
        return X.to(self.gpu_id), y.to(self.gpu_id)

    def _train_epoch(self) -> dict[str, Any]:
        
        self.model.train()

        loss_metric = RunningMetric()
        collector = StatsCollector()

        loader = tqdm(self.loaders["train"], total = len(self.loaders["train"]), desc= f"Training, epoch {self.epochs_trained}")

        for X, y in loader:

            #Batch iteration
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]

            self.optimizer.zero_grad()
            y_pred = self.forward(X)

            loss = self.loss_fn(y_pred, y)
            loss.backward()

            self.optimizer.step()

            loss_metric.update(loss.item(), batch_size)
            collector.update(y_pred, y)
            
            loader.set_postfix(
                loss = loss_metric.average)

        y_pred_all = collector.collect_y_pred() #(N, C)
        y_pred_argmax = y_pred_all.argmax(axis=1) #(N,)
        y_true_all = collector.collect_y_true()

        mae = mean_absolute_error(y_true_all, y_pred_argmax)
        accuracy = accuracy_score(y_true_all, y_pred_argmax)
        cost = calc_cost_metric(y_true=y_true_all, y_pred=y_pred_argmax, n_classes=self.num_classes)
       
        distribution = np.bincount(y_pred_argmax, minlength=self.num_classes)
        distribution = distribution/distribution.sum()

        self.epochs_trained +=1

        results = {
            "train_distribution": distribution, #numpy array
            "train_loss": loss_metric.average, #single value
            "train_accuracy":accuracy, #single value
            "train_mae": mae,
            "train_cost": cost
            }

        return results
        
    @torch.no_grad()
    def _eval_epoch(self, phase) -> dict[str, Any]:

        self.model.eval()

        loss_metric = RunningMetric()        
        collector = StatsCollector()

        loader = self.loaders[phase]

        for X, y in loader:
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]
            y_pred = self.forward(X)
            loss = self.loss_fn(y_pred, y)

            loss_metric.update(loss.item(), batch_size)
            collector.update(y_pred, y)
        
        y_pred_all = collector.collect_y_pred()
        y_pred_argmax = y_pred_all.argmax(axis=1)
        y_true_all = collector.collect_y_true()

        if phase =="test":
            self.log_predictions(y_pred_all)
        
        #Some metrics
        mae = mean_absolute_error(y_true_all, y_pred_argmax)
        accuracy = accuracy_score(y_true_all, y_pred_argmax)
        cost = calc_cost_metric(y_true=y_true_all, y_pred=y_pred_argmax, n_classes=self.num_classes)
       
        distribution = np.bincount(y_pred_argmax, minlength=self.num_classes)
        distribution = distribution/distribution.sum()

        results = {
            f"{phase}_distribution": distribution, #numpy array
            f"{phase}_loss": loss_metric.average, #single value
            f"{phase}_accuracy":accuracy, #single value
            f"{phase}_mae": mae, #single value
            f"{phase}_cost": cost, #single value
            }
            
        return results
    
    def log_predictions(self, y_pred_all):
        my_df = pd.DataFrame(y_pred_all)
        path = f"{self.model_id}_preds_{self.epochs_trained}.csv"
        my_df.to_csv(path)
        mlflow.log_artifact(path)
        os.remove(path)

    def train_until_converge(self, n_epochs, patience, min_delta, sch_stepsize, sch_gamma) -> None:

        early_stopper = EarlyStopper(
            patience=patience, 
            min_delta=min_delta)
        
        scheduler = StepLR(self.optimizer, step_size=sch_stepsize, gamma=sch_gamma, verbose=True)

        for _ in range(n_epochs):
            train_results = self._train_epoch()
            scheduler.step()
            val_results = self._eval_epoch("val")

            mlflow.log_metrics(get_only_metrics(val_results), step = self.epochs_trained)
            mlflow.log_metrics(get_only_metrics(train_results), step = self.epochs_trained)

            early_stopper.step(val_results["val_loss"]) #One more step for validation loss, check whether should stop.

            if early_stopper.is_best_model:
                #This is the best model so far, let's save it.
                self._save_checkpoint()

            if early_stopper.early_stop:
                break #Model converged.

        print(f"Model Converged! the best validation loss is {early_stopper.min_validation_loss}")
        self._load_checkpoint()
        os.remove(self.checkpoint_path)
    
    def _save_checkpoint(self):

        ckp = {
                "epoch": self.epochs_trained,
                "model_state_dict":self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
              }

        torch.save(ckp, self.checkpoint_path)
        mlflow.log_artifact(local_path=self.checkpoint_path)
        
    def _load_checkpoint(self):
        ckp = torch.load(self.checkpoint_path)
        self.epochs_trained = ckp["epoch"]
        self.model.load_state_dict(ckp["model_state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])

    def set_loss_fn(self, loss_fn:nn.Module):
        self.loss_fn = loss_fn
        self.loss_fn.to(self.gpu_id)

class SingleGPUTrainerMatan:
    def __init__(
        self, 
        model: nn.Module, 
        loaders: Dict[str, DataLoader],
        optimizer: torch.optim.Optimizer, 
        gpu_id: int,
        save_every: int,
        num_classes: int,
        grad_norm:float = 15.0
        ):

        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.loaders = loaders

        self.optimizer = optimizer
        self.save_every = save_every
        self.num_classes = num_classes
        self.epochs_trained = 0
        
        self.checkpoint_path = Path("models", f"{uuid.uuid4().hex}.pt")
        self.grad_norm = grad_norm

    def forward(self, X):
        return F.softmax(self.model(X), dim = 1) #Normalized        

    def prepare_input(self, X, y):
        return X.to(self.gpu_id), y.to(self.gpu_id)

    def _train_epoch(self) -> dict[str, Any]:
        
        self.model.train()

        loss_metric = RunningMetric()
        collector = StatsCollector()

        loader = tqdm(self.loaders["train"], total = len(self.loaders["train"]), desc= f"Training, epoch {self.epochs_trained}")

        for X, y in loader:

            #Batch iteration
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]

            self.optimizer.zero_grad()
            y_pred = self.forward(X)

            loss = self.loss_fn(y_pred, y)
            loss.backward()

            self.optimizer.step()

            loss_metric.update(loss.item(), batch_size)
            collector.update(y_pred, y)
            
            loader.set_postfix(
                loss = loss_metric.average)
        
        self.epochs_trained +=1

        y_pred_all = collector.collect_y_pred().argmax(axis=1)
        y_true_all = collector.collect_y_true()

        ones_ratio =  y_pred_all.mean()
        accuracy = accuracy_score(y_true_all, y_pred_all)

        results = {
            "train_loss": loss_metric.average, #single value
            "train_accuracy":accuracy, #single value
            "train_ones_ratio":ones_ratio, #single value
            }

        return results
        
    @torch.no_grad()
    def _eval_epoch(self, phase) -> dict[str, Any]:

        self.model.eval()

        loss_metric = RunningMetric()        
        collector = StatsCollector()

        loader = self.loaders[phase]

        for X, y in loader:
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]
            y_pred = self.forward(X)
            loss = self.loss_fn(y_pred, y)

            loss_metric.update(loss.item(), batch_size)
            collector.update(y_pred, y)
        
        y_pred_all = collector.collect_y_pred().argmax(axis=1) #Binary vector of 0 and 1s
        y_true_all = collector.collect_y_true()
        
        #Some metrics
        
        ones_ratio =  y_pred_all.mean()
        accuracy = accuracy_score(y_true_all, y_pred_all)

        results = {
            f"{phase}_loss": loss_metric.average, #single value
            f"{phase}_accuracy":accuracy, #single value
            f"{phase}_ones_ratio":ones_ratio, #single value
            }

        return results

    def train_until_converge(
        self, n_epochs:int, 
        patience:int=3, min_delta:float = 1.0, 
        sch_stepsize:int=5, sch_gamma:float=0.9) -> None:

        early_stopper = EarlyStopper(
            patience=patience, 
            min_delta=min_delta)
        
        scheduler = StepLR(self.optimizer, step_size=sch_stepsize, gamma=sch_gamma, verbose=True)

        for _ in range(n_epochs):
            train_results = self._train_epoch()
            scheduler.step()
            val_results = self._eval_epoch(phase ="val")

            mlflow.log_metrics(get_only_metrics(val_results), step = self.epochs_trained)
            mlflow.log_metrics(get_only_metrics(train_results), step = self.epochs_trained)

            early_stopper.step(val_results["val_loss"]) #One more step for validation loss, check whether should stop.

            if early_stopper.is_best_model:
                #This is the best model so far, let's save it.
                
                best_epoch_idx = self.epochs_trained
                self._save_checkpoint()

            if early_stopper.early_stop:
                break #Model converged.

        print(f"Model Converged! the best validation loss is {early_stopper.min_validation_loss}")
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.epochs_trained = best_epoch_idx
    
    def _save_checkpoint(self):
        ckp = self.model.state_dict()
        torch.save(ckp, self.checkpoint_path)

    def set_loss_fn(self, loss_fn:nn.Module):
        self.loss_fn = loss_fn
        self.loss_fn.to(self.gpu_id)