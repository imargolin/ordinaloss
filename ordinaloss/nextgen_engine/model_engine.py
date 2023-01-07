from ordinaloss.utils.metric_utils import RunningMetric, BinCounter
from ordinaloss.utils.metric_utils import accuracy_pytorch, mae_pytorch
import torch.nn.functional as F
from tqdm import tqdm
import torch
import mlflow
import numpy as np

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

class OrdinalEngine:
    def __init__(self, model, loss_fn, device, loaders, 
                optimizer_fn, n_classes, use_lr_scheduler=False,
                scheduler_lambda_fn=None, callbacks=[],  **optimizer_params):

        #Setting the device        
        self.device = device

        #Setting the model
        self.model = model
        self.model.to(self.device)

        self.loaders = loaders

        #self.set_loss_fn(loss_fn)
        
        self.optimizer_params = optimizer_params
        self.optimizer_fn = optimizer_fn


        self.set_optimizer(optimizer_fn, **self.optimizer_params)
        self.epochs_trained = 0
        self.n_classes = n_classes

        self.callbacks = callbacks

        for callback in self.callbacks:
            callback.on_init(self)


        #self.init_mlrun()

    def init_mlrun(self):
        mlflow.end_run() #Ending run if exists.

        experiment_names = [experiment.name for experiment in mlflow.list_experiments()]
        if self.__class__.__name__ not in experiment_names:
            mlflow.create_experiment(self.__class__.__name__)

        experiment_id = mlflow.get_experiment_by_name(self.__class__.__name__).experiment_id
        mlflow.start_run(experiment_id=experiment_id)

        mlflow.log_params(self.optimizer_params)
        mlflow.log_param("optimizer_fn", self.optimizer_fn.__name__)
        #mlflow.log_param("lr_scheduler", self.use_lr_scheduler)
        mlflow.log_param("loss_fn", self._loss_fn.__repr__())
        mlflow.log_param("model_name", self.model._get_name())
        mlflow.log_param("n_layers", len(list(self.model.parameters())))

    def set_loss_fn(self, loss_fn):
        self._loss_fn = loss_fn
        self._loss_fn.to(self.device)

    def set_optimizer(self, optimizer_fn, **optimizer_params):
        '''
        Setting an optimizer, happens in the __init__
        '''
        
        self._optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        mlflow.log_params(optimizer_params)

    def forward(self, X):
        return F.softmax(self.model(X), dim = 1) #Normalized        

    def prepare_input(self, X, y):
        return X.to(self.device), y.to(self.device)

    def _train_epoch(self):
        
        self.model.train()

        loader = self.loaders["train"]
        accuracy_metric = RunningMetric()
        loss_metric = RunningMetric()
        mae_metric = RunningMetric()
        bin_counter = BinCounter(n_classes = self.n_classes, device=self.device)

        iterator = tqdm(loader, total = len(loader), desc= f"Training, epoch {self.epochs_trained}")

        for callback in self.callbacks:
            callback.on_train_start(self)

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
            mae = mae_pytorch(y_pred, y)

            mae_metric.update(mae, batch_size)
            accuracy_metric.update(accuracy, batch_size)
            loss_metric.update(loss.item(), batch_size)
            bin_counter.update(y_pred.argmax(axis=1)) 
            
            iterator.set_postfix(loss = loss_metric.average, 
                                 accuracy = accuracy_metric.average, 
                                 mae = mae_metric.average)

        for callback in self.callbacks:
            callback.on_train_end(self)

        self.epochs_trained+=1

    @torch.no_grad()
    def _eval_epoch(self, phase = "val"):
        self.model.eval()
        loader = self.loaders[phase]

        accuracy_metric = RunningMetric()
        loss_metric = RunningMetric()
        bin_counter = BinCounter(n_classes = self.n_classes, device=self.device)

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
                #TODO:
                #if satisfy constraints:
                    #break
                #else:
                    #modify loss and iterate all_over.
                #Eval on testset (satisfies constraints?)
                #Modify loss and start retrain

    # def satisfy_constraints(self, constraints):
        
    #     constraints = torch.tensor(constraints, device = self.device)
    #     test_dist = self.predict_dist_on_test()

    #     return (test_dist < constraints).all().item()

    def predict_dist_on_test(self, phase="test"):
        self.model.eval()
        loader = self.loaders[phase]
        bin_counter = BinCounter(n_classes = self.n_classes, device=self.device)
        for X, y in loader:
            X, y = self.prepare_input(X, y)
            batch_size = y.shape[0]
            y_pred = self.forward(X)

            bin_counter.update(y_pred.argmax(axis=1))

        return bin_counter.average


        



