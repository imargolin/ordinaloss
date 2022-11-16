import torch
from dataclasses import dataclass, field

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

@torch.no_grad()
def accuracy_pytorch(y_pred, y):
    y_arg_pred = y_pred.argmax(axis=1)
    return (y_arg_pred == y).to(torch.float32).mean().item()
