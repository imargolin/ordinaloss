from .metrics import RunningMetric, BinCounter
from .metrics import accuracy_pytorch
import torch.nn.functional as F
from tqdm import tqdm
import torch


print(f"loaded {__name__}")



class OrdinalEngine:
    def __init__(self, model, loss_fn, device, loaders, 
                optimizer_fn, n_classes, use_lr_scheduler=False,
                scheduler_lambda_fn=None, **optimizer_params):

        #Setting the device        
        self.device = device

        #Setting the model
        self.model = model
        self.model.to(self.device)

        self.loaders = loaders

        self.set_loss_fn(loss_fn)
        self.set_optimizer(optimizer_fn, **optimizer_params)
        self.epochs_trained = 0
        self.n_classes = n_classes


    def set_loss_fn(self, loss_fn):
        self._loss_fn = loss_fn

    def set_optimizer(self, optimizer_fn, **optimizer_params):
        '''
        Setting an optimizer, happens in the __init__
        '''
        
        self._optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)

    def forward(self, X):
        return F.softmax(self.model(X), dim = 1) #Normalized        

    def prepare_input(self, X, y):
        return X.to(self.device), y.to(self.device)

    def _train_epoch(self):
        
        self.model.train()

        loader = self.loaders["train"]
        accuracy_metric = RunningMetric()
        loss_metric = RunningMetric()
        bin_counter = BinCounter(n_classes = self.n_classes)

        iterator = tqdm(loader, total = len(loader), desc= f"Training, epoch {self.epochs_trained}")
        for X, y in iterator:

            #Batch iteration
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]

            self._optimizer.zero_grad()
            y_pred = self.forward(X)

            loss = self._loss_fn(y_pred, y)
            loss.backward()

            self._optimizer.step()

            accuracy = accuracy_pytorch(y_pred, y)

            accuracy_metric.update(accuracy, batch_size)
            loss_metric.update(loss.item(), batch_size)
            bin_counter.update(y_pred.argmax(axis=1).bincount(minlength=self.n_classes)) 
            
            iterator.set_postfix(loss = loss_metric.average, 
                                 accuracy = accuracy_metric.average) 
            
            print(bin_counter.values)

        self._on_epoch_end()


    @torch.no_grad()
    def _eval_epoch(self, phase = "val"):
        self.model.eval()
        loader = self.loaders[phase]

        accuracy_metric = RunningMetric()
        loss_metric = RunningMetric()
        bin_counter = BinCounter(n_classes = self.n_classes)

        iterator = tqdm(loader, total = len(loader), desc= f"Evaluating...")
        for X, y in iterator:
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]
            y_pred = self.forward(X)
            loss = self._loss_fn(y_pred, y)
            accuracy = accuracy_pytorch(y_pred, y)

            accuracy_metric.update(accuracy, batch_size)
            loss_metric.update(loss.item(), batch_size)
            bin_counter.update(y_pred.argmax(axis=1)) 

            iterator.set_postfix(loss = loss_metric.average, 
                                 accuracy = accuracy_metric.average) 

        print(bin_counter.average)
        print(loss_metric.average)
        print(accuracy_metric.average)






    
    def _on_epoch_end(self):
        


        pass



