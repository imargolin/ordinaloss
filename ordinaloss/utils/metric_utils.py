import torch
from dataclasses import dataclass, field
import numpy as np

print(f"loaded {__name__}!!")

@dataclass
class RunningMetric:
    values: list[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0
    
    def update(self, value:float, batch_size: int):
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates

class BinCounter:
    def __init__(self, n_classes, device):
        self.device = device
        self.running_total = torch.zeros(n_classes, device=self.device)
        self.num_updates = 0
        self.n_classes = n_classes
    
    @torch.no_grad()
    def update(self, y_pred):
        
        #y_pred after argmax
        counts = y_pred.bincount(minlength=self.n_classes)
        self.running_total = self.running_total + counts 
        self.num_updates += y_pred.shape[0] #batch_size
        self.average = self.running_total / self.num_updates
        
class PredictionsCollector:
    def __init__(self, n_classes, device):
        self.device = device
        self.predictions_collector = []
        self.actual_collector = []
        
    @torch.no_grad()
    def update(self, y_pred:torch.Tensor, y_true: torch.Tensor):
        '''
        y_pred is NxC (even if C is 2)
        y_true is N (actual classes [1,2,3,0,2...])
        
        '''
        self.predictions_collector.append(y_pred.cpu().numpy())
        self.actual_collector.append(y_true.cpu().numpy())
        
    def finalize(self):
        self.predictions_collector = np.concatenate(self.predictions_collector) #should be numpy
        self.actual_collector = np.concatenate(self.actual_collector)    
    

@torch.no_grad()
def accuracy_pytorch(y_pred, y):
    y_arg_pred = y_pred.argmax(axis=1)
    return (y_arg_pred == y).to(torch.float32).mean().item()

def mae_pytorch(y_pred, y):
    y_arg_pred = y_pred.argmax(axis=1).to(torch.float32)
    return (y_arg_pred - y).abs().to(torch.float32).mean().item()
