import torch
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import confusion_matrix

print(f"loaded {__name__}")

def calc_cost_metric(y_pred, y_true, n_classes=5):
    """_summary_

    Args:
        y_pred (_type_): n numpy (1 dim, predictions after argmax)
        y_test (_type_): n numpy (1 dim)
    """

    r = np.arange(0,n_classes)
    cost_matrix = 2 * np.abs(r-r[:, None])+1
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels = np.arange(0,n_classes))
    return (cm * cost_matrix).sum()

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

class StatsCollector:
    """
    Used to aggregate data, the update method allow us to keep in memory the predictions of the entire epoch.
    """
    def __init__(self):
        self.y_pred = []
        self.y_true = []
    
    @torch.no_grad()
    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred.cpu().numpy())
        self.y_true.append(y_true.cpu().numpy())

    def collect_y_pred(self) -> np.array:
        return np.concatenate(self.y_pred)
        
    def collect_y_true(self)-> np.array:
        return np.concatenate(self.y_true, axis=-1)

#Not so used
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


@torch.no_grad()
def accuracy_pytorch(y_pred, y):
    y_arg_pred = y_pred.argmax(axis=1)
    return (y_arg_pred == y).to(torch.float32).mean().item()

@torch.no_grad()
def mae_pytorch(y_pred, y):
    y_arg_pred = y_pred.argmax(axis=1).to(torch.float32)
    return (y_arg_pred - y).abs().to(torch.float32).mean().item()
