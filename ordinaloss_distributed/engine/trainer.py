from ordinaloss.utils.metric_utils import RunningMetric, BinCounter
from ordinaloss.utils.metric_utils import accuracy_pytorch, mae_pytorch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm
import torch
import mlflow
import numpy as np

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

print(f"loaded {__name__} ")


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.99):
        self.patience = patience
        self.min_delta = min_delta #We want the loss to be 0.99 from the previous epoch.
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        print(self.min_validation_loss)
        print(validation_loss)

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            
        elif validation_loss > (self.min_validation_loss * self.min_delta):
            print(f"strike {self.counter}! didn't increase by mindelta.")
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_data: DataLoader, 
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer, 
        gpu_id: int,
        save_every: int,
    ):

        #Setting the device        
        self.gpu_id = gpu_id
        self.is_main = self.gpu_id == 0

        #Setting the model
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.epochs_trained = 0


    def forward(self, X):
        return F.softmax(self.model(X), dim = 1) #Normalized        

    def prepare_input(self, X, y):
        return X.to(self.gpu_id), y.to(self.gpu_id)

    def _train_epoch(self, epoch):
        
        self.model.train()

        accuracy_metric = RunningMetric()
        loss_metric = RunningMetric()
        mae_metric = RunningMetric()
        bin_counter = BinCounter(n_classes = 5, device=self.gpu_id)

        loader = self.train_data

        if self.is_main:
            #progress bar for the first gpu
            loader = tqdm(loader, total = len(loader), desc= f"Training, epoch {epoch}")

        for X, y in loader:

            #Batch iteration
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]

            self.optimizer.zero_grad()
            y_pred = self.forward(X)

            loss = self.loss_fn(y_pred, y)
            loss.backward()

            self.optimizer.step()

            accuracy = accuracy_pytorch(y_pred, y)
            mae = mae_pytorch(y_pred, y)

            mae_metric.update(mae, batch_size)
            accuracy_metric.update(accuracy, batch_size)
            loss_metric.update(loss.item(), batch_size)
            bin_counter.update(y_pred.argmax(axis=1)) 
            
            if self.is_main:

                loader.set_postfix(loss = loss_metric.average, 
                                    accuracy = accuracy_metric.average, 
                                    mae = mae_metric.average)
  
    def _save_checkpoint(self, epoch):
        if self.is_main:
            ckp = self.model.module.state_dict()
            torch.save(ckp, "checkpoint.pt")
            print(f"Epoch: {epoch} | Training checkpoint saved at checkpoint.pt")

        pass

    @torch.no_grad()
    def _eval_epoch(self):
        if True:
            self.model.eval()
            loader = self.validation_data

            accuracy_metric = RunningMetric()
            loss_metric = RunningMetric()
            bin_counter = BinCounter(n_classes = 5, device=self.gpu_id)

            loader = tqdm(loader, total = len(loader), desc= f"Evaluating...")
            for X, y in loader:
                X, y = self.prepare_input(X, y)
                batch_size = y.shape[0]
                y_pred = self.forward(X)
                loss = self.loss_fn(y_pred, y)
                accuracy = accuracy_pytorch(y_pred, y)

                accuracy_metric.update(accuracy, batch_size)
                loss_metric.update(loss.item(), batch_size)
                bin_counter.update(y_pred.argmax(axis=1)) 

                loader.set_postfix(
                    loss = loss_metric.average, 
                    accuracy = accuracy_metric.average
                    ) 

            return (bin_counter.average, 
                    loss_metric.average, 
                    accuracy_metric.average)

    def train_until_converge(self, n_epochs, patience, min_delta):

        early_stopper = EarlyStopper(patience = patience, 
                                     min_delta=min_delta)

        for i in range(n_epochs):
            self._train_epoch()
            _, eval_loss, eval_accuracy = self._eval_epoch()
            if early_stopper.early_stop(eval_loss):
                print("model converges")
                break #Model converged.

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        self.loss_fn.to(self.gpu_id)

    def predict_dist_on_test(self, phase="test"):
        self.model.eval()
        loader = self.loaders[phase]
        bin_counter = BinCounter(n_classes = self.n_classes, device=self.gpu_id)
        for X, y in loader:
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]
            y_pred = self.forward(X)

            bin_counter.update(y_pred.argmax(axis=1))

        return bin_counter.average


        



