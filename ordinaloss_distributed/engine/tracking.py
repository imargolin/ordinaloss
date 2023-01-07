from abc import ABC
from enum import Enum, auto

class Stage(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    AUTO_TEST = auto()


class ExperimentTracker(ABC):
    pass
    
print("---")

print(Stage.TRAIN)
print(Stage.TRAIN.name)
