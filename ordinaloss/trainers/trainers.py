
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
from sklearn.metrics import accuracy_score, mean_absolute_error,f1_score,precision_score,recall_score,roc_auc_score
import os
import uuid
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

class SingleGPUTrainerMatan:
    def __init__(
        self, 
        model: nn.Module, 
        loaders: Dict[str, DataLoader],
        optimizer: torch.optim.Optimizer, 
        gpu_id: int,
        save_every: int,
        num_classes: int,
        grad_norm:float = 15.0,
        opt_metric:str = 'val_loss'
        ):

        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.loaders = loaders

        self.optimizer = optimizer
        self.num_classes = num_classes
        self.epochs_trained = 0
        
        self.model_id = secrets.token_hex(nbytes=16)
        self.checkpoint_path = f"{self.model_id}.pt"

        self.grad_norm = grad_norm
        self.opt_metric = opt_metric
        
    def forward(self, X):
        return F.softmax(self.model(X), dim = 1) #Normalized        

    def prepare_input(self, X, y):
        return X.to(self.gpu_id), y.to(self.gpu_id)

    def log_results(self, df:pd.DataFrame):
        path = f"{self.model_id}_results_{self.epochs_trained}.csv"
        
        df.to_csv(path)
        mlflow.log_artifact(path)
        os.remove(path)

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

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 6)

            self.optimizer.step()

            loss_metric.update(loss.item(), batch_size)
            collector.update(y_pred, y)
            
            loader.set_postfix(
                loss = loss_metric.average)
        
        self.epochs_trained +=1

        y_pred_all = collector.collect_y_pred()
        y_pred_all_argmax = y_pred_all.argmax(axis=1)
        y_true_all = collector.collect_y_true()

        ones_ratio =  y_pred_all_argmax.mean()
        accuracy = accuracy_score(y_true_all, y_pred_all_argmax)
        f1 = f1_score(y_true_all, y_pred_all_argmax)
        precision = precision_score(y_true_all, y_pred_all_argmax)
        recall = recall_score(y_true_all, y_pred_all_argmax)
        roc_auc = roc_auc_score(y_true_all, y_pred_all_argmax)

        results = {
            "train_loss": loss_metric.average, #single value
            "train_accuracy":accuracy, #single value
            "train_ones_ratio":ones_ratio, #single value
            'train_f1_score' : f1,
            'train_precision_score' : precision,
            'train_recall_score' : recall,
            'train_roc_auc_score' : roc_auc            
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
        
        y_pred_all = collector.collect_y_pred() #Nx2
        y_pred_all_argmax = y_pred_all.argmax(axis=1)
        y_true_all = collector.collect_y_true()
        
        #Some metrics
        
        ones_ratio =  y_pred_all_argmax.mean()
        accuracy = accuracy_score(y_true_all, y_pred_all_argmax)
        f1 = f1_score(y_true_all, y_pred_all_argmax)
        precision = precision_score(y_true_all, y_pred_all_argmax)
        recall = recall_score(y_true_all, y_pred_all_argmax)
        roc_auc = roc_auc_score(y_true_all, y_pred_all_argmax)
        
        results = {
            f"{phase}_loss": loss_metric.average, #single value
            f"{phase}_accuracy":accuracy, #single value
            f"{phase}_ones_ratio":ones_ratio, #single value
            f"{phase}_f1_score":f1, #single value
            f"{phase}_precision_score":precision, #single value
            f"{phase}_recall_score":recall, #single value
            f"{phase}_roc_auc_score":roc_auc, #single value
            f"{phase}_y_pred": y_pred_all, #single value
            f"{phase}_y_true": y_true_all #single value
            }

        if phase =="test":
            df = pd.DataFrame({
                "y_pred":y_pred_all[:,1], 
                "y_true":y_true_all}
                )
                
            self.log_results(df)


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
            
            if 'loss' in self.opt_metric:
                early_stopper.step(val_results[self.opt_metric]) #One more step for validation loss, check whether should stop.
            else:
                early_stopper.step(-1 * val_results[self.opt_metric]) #One more step for validation loss, check whether should stop.

            if early_stopper.is_best_model:
                self._save_checkpoint() #This is the best model so far, let's save it.

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

        #checkpoint path is a temporary path, and saved in mlflow shit.
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