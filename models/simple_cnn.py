import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4, filters=32, kernel_size=3, p_dropout: float = 0.0, input_shape=(1, 28, 28)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()


        with torch.no_grad():
            x = torch.zeros(1, *input_shape) 
            x = self._forward_features(x)
            flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
