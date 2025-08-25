import os
import random
import numpy as np
import torch
import torch.nn as nn

from utils.dataset import DatasetLoader
from search_algorithms.ga_search import GeneticSearch, make_optimizer
from search_algorithms.grid_search import GridSearch
from search_algorithms.random_search import RandomSearch
from search_algorithms.bayesian_search import BayesianOptSearch
from models.simple_cnn import SimpleCNN
from trainer import Trainer


# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === CONFIG ===
DATA_DIR = r"C:\\Users\\w\\Desktop\\Kodlama\\VsCode\\HelloWorld\\DroneDetection\\file8_lastversion"
FITNESS_EPOCHS = 2
BALANCED_SAMPLING = False

param_space = {
    "lr": [5e-4, 1e-3, 5e-3],
    "batch_size": [16, 32, 64],
    "optimizer": ["Adam", "SGD", "RMSprop"],
    "filters": [16, 32, 64],
    "kernel_size": [3, 5],
    "dropout": [0.0, 0.2, 0.5],
    "weight_decay": [0.0, 1e-4, 1e-3],
}


def train_full(best_h, epochs=10):
    ds = DatasetLoader(DATA_DIR)
    train_loader, val_loader, test_loader = ds.make_loaders(best_h['batch_size'], balanced=BALANCED_SAMPLING, device=device)
    model = SimpleCNN(num_classes=ds.num_classes,
                      filters=best_h['filters'], kernel_size=best_h['kernel_size'],
                      p_dropout=best_h.get('dropout', 0.0)).to(device)
    optimizer = make_optimizer(best_h['optimizer'], model.parameters(), best_h['lr'], weight_decay=best_h.get('weight_decay', 0.0))
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, train_loader, val_loader, optimizer, device, criterion)

    best_val_acc = -1.0
    best_path = "best_simplecnn.pth"

    for ep in range(1, epochs + 1):
        tr_loss = trainer.train_one_epoch()
        v_loss, v_acc, v_p, v_r, v_f1 = trainer.eval_metrics()
        print(f"Epoch {ep:02d} | TrainLoss {tr_loss:.4f} | ValLoss {v_loss:.4f} | ValAcc {v_acc*100:.2f}% | ValPrec {v_p*100:.2f}% | ValRec {v_r*100:.2f}% | ValF1 {v_f1*100:.2f}%")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)

    # Test
    print("\n=== Test Evaluation ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    trainer = Trainer(model, train_loader, test_loader, optimizer, device, criterion)
    t_loss, t_acc, t_p, t_r, t_f1 = trainer.eval_metrics()
    print(f"TEST -> Loss: {t_loss:.4f} | Acc: {t_acc*100:.2f}% | Precision: {t_p*100:.2f}% | Recall: {t_r*100:.2f}% | F1: {t_f1*100:.2f}%")


if __name__ == "__main__":
    ds = DatasetLoader(DATA_DIR)

    # === GA ===
    ga = GeneticSearch(param_space, ds, device, population_size=6, generations=5, mutation_rate=0.30, fitness_epochs=FITNESS_EPOCHS, balanced=BALANCED_SAMPLING)
    best_ga, _ = ga.search()

    # === Grid ===
    grid = GridSearch(param_space, ds, device, fitness_epochs=FITNESS_EPOCHS, balanced=BALANCED_SAMPLING)
    best_grid, _ = grid.search()

    # === Random ===
    rand = RandomSearch(param_space, ds, device, n_trials=10, fitness_epochs=FITNESS_EPOCHS, balanced=BALANCED_SAMPLING)
    best_rand, _ = rand.search()

    # ===
