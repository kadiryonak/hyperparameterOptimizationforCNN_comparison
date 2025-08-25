import random
import torch.optim as optim
from .base_search import HyperparameterSearch
from trainer import Trainer
from models.simple_cnn import SimpleCNN
import torch.nn as nn


def make_optimizer(name, params, lr, weight_decay=0.0):
    if name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'SGD':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'RMSprop':
        return optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


class GeneticSearch(HyperparameterSearch):
    def __init__(self, param_space, dataset_loader, device, population_size=6, generations=5, mutation_rate=0.3, fitness_epochs=1, balanced=False):
        super().__init__(param_space, dataset_loader, device)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_epochs = max(2, fitness_epochs) 
        self.balanced = balanced
        self.criterion = nn.CrossEntropyLoss()

    def _sample_from_space(self):
        return {k: random.choice(v) for k, v in self.param_space.items()}

    def _mutate(self, ind: dict):
        for k in ind.keys():
            if random.random() < self.mutation_rate:
                ind[k] = random.choice(self.param_space[k])
        return ind

    def _crossover(self, p1: dict, p2: dict):
        return {k: (p1[k] if random.random() < 0.5 else p2[k]) for k in p1.keys()}

    def _build_trainer(self, h: dict) -> Trainer:
        
        train_loader, val_loader, _ = self.dataset_loader.make_loaders(h['batch_size'], balanced=self.balanced, device=self.device)

        model = SimpleCNN(num_classes=self.dataset_loader.num_classes, use_bn=h.get('use_bn', False), p_dropout=h.get('dropout', 0.0))
        opt = make_optimizer(h['optimizer'], model.parameters(), h['lr'], weight_decay=h.get('weight_decay', 0.0))
        return Trainer(model, train_loader, val_loader, opt, self.device, self.criterion)

    def _evaluate(self, h: dict) -> float:
        trainer = self._build_trainer(h)
        
        acc = trainer.fit(epochs=self.fitness_epochs, verbose=False)
        return acc

    def search(self):
        population = [self._sample_from_space() for _ in range(self.population_size)]
        best_h, best_acc = None, -1.0

        for gen in range(self.generations):
            print(f"\n=== GA Generation {gen+1}/{self.generations} ===")
            scored = []
            for ind in population:
                acc = self._evaluate(ind)
                scored.append((acc, ind))
                print(f"Individual: {ind} -> ValAcc {acc*100:.2f}%")

            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > best_acc:
                best_acc, best_h = scored[0]

            # elitizm
            new_pop = [scored[0][1].copy(), scored[1][1].copy()]
            # crossover + mutation
            pool = [h for _, h in scored[:min(4, len(scored))]]
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(pool, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            population = new_pop

        print(f"\nBest Individual: {best_h} | Best ValAcc: {best_acc*100:.2f}%")
        return best_h, best_acc

