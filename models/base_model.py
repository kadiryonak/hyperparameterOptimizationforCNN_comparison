# ──────────────────────────────────────────────────────────────────────────────
# file: base_model.py
# ──────────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    def __init__(self, num_classes=4, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

    @abstractmethod
    def build_model(self):
        """Modeli oluştur ve return et"""
        pass
