# ──────────────────────────────────────────────────────────────────────────────
# file: simple_cnn_model.py
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel

class SimpleCNNModel(BaseModel):
    def __init__(self, num_classes=4, dropout=0.0, input_shape=(1,28,28), filters=32, kernel_size=3):
        super().__init__(num_classes, dropout)
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.model = self.build_model()

    def build_model(self):
        conv1 = nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, padding=1)
        conv2 = nn.Conv2d(self.filters, self.filters*2, kernel_size=self.kernel_size, padding=1)
        pool = nn.MaxPool2d(2,2)
        drop = nn.Dropout(self.dropout) if self.dropout>0 else nn.Identity()

        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = F.relu(conv1(x))
            x = pool(F.relu(conv2(x)))
            x = pool(x)
            x = drop(x)
            flattened_size = x.view(1, -1).size(1)

        fc1 = nn.Linear(flattened_size, 128)
        fc2 = nn.Linear(128, self.num_classes)

        return nn.Sequential(conv1, conv2, pool, drop, nn.Flatten(), fc1, nn.ReLU(), fc2)

    def forward(self, x):
        return self.model(x)
