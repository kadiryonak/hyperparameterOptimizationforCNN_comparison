
from abc import ABC, abstractmethod


class HyperparameterSearch(ABC):
    def __init__(self, param_space, dataset_loader, device):
        self.param_space = param_space
        self.dataset_loader = dataset_loader
        self.device = device

    @abstractmethod
    def search(self):
        pass