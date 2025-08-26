import itertools
import torch.nn as nn
from .base_search import HyperparameterSearch
from trainer import Trainer
from models.simple_cnn import SimpleCNN
from .ga_search import make_optimizer


class GridSearch(HyperparameterSearch):
    def __init__(self, param_space, dataset_loader, device, fitness_epochs=2, balanced=False):
        super().__init__(param_space, dataset_loader, device)
        self.fitness_epochs = fitness_epochs
        self.balanced = balanced
        self.criterion = nn.CrossEntropyLoss()

    def search(self):
        keys, values = zip(*self.param_space.items())
        best_h, best_acc = None, -1.0
        for combo in itertools.product(*values):
            h = dict(zip(keys, combo))
            train_loader, val_loader, _ = self.dataset_loader.make_loaders(h['batch_size'], balanced=self.balanced, device=self.device)
            model = SimpleCNN(num_classes=self.dataset_loader.num_classes,
                              filters=h['filters'], kernel_size=h['kernel_size'],
                              p_dropout=h.get('dropout',0.0))
            opt = make_optimizer(h['optimizer'], model.parameters(), h['lr'], weight_decay=h['weight_decay'])
            trainer = Trainer(model, train_loader, val_loader, opt, self.device, self.criterion)
            acc = trainer.fit(epochs=self.fitness_epochs)
            print(f"Grid {h} -> Acc {acc*100:.2f}%")
            if acc > best_acc:
                best_h, best_acc = h, acc
        return best_h, best_acc
