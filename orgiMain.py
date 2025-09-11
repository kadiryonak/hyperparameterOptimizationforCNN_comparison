import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from utils.dataset import DatasetLoader
from search_algorithms.ga_search import GeneticSearch, make_optimizer
from trainer import Trainer
from models.simple_cnn import SimpleCNN
from models.base_model import BaseResNet  # abstract class ile ResNet tanımı

# -----------------------------
# Reproducibility
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = r"***\\DroneDetection\\file8_lastversion"
FITNESS_EPOCHS = 2
BALANCED_SAMPLING = False

# Optimize edilecek parametreler
param_space = {
    "lr": [1e-5, 1e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "dropout": [0.1, 0.3, 0.5]
}

# Sabit parametreler
FIXED_PARAMS = {
    "filters": 32,
    "kernel_size": 3,
    "resnet_fc_hidden": 256
}

# -----------------------------
# Full training with best hyperparams
# -----------------------------
def train_full(model_name, best_h, epochs=10):
    ds = DatasetLoader(DATA_DIR)
    train_loader, val_loader, test_loader = ds.make_loaders(
        best_h['batch_size'], balanced=BALANCED_SAMPLING, device=device
    )

    if model_name == "SimpleCNN":
        model = SimpleCNN(
            num_classes=ds.num_classes,
            filters=FIXED_PARAMS['filters'],
            kernel_size=FIXED_PARAMS['kernel_size'],
            p_dropout=best_h.get('dropout', 0.0)
        ).to(device)
    elif model_name == "ResNet18":
        model = BaseResNet(
            num_classes=ds.num_classes,
            fc_hidden=FIXED_PARAMS['resnet_fc_hidden'],
            p_dropout=best_h.get('dropout', 0.0)
        ).to(device)
    else:
        raise ValueError("Model must be either 'SimpleCNN' or 'ResNet18'")

    optimizer = make_optimizer(
        "Adam", model.parameters(), best_h['lr'], weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, train_loader, val_loader, optimizer, device, criterion)

    best_val_acc = -1.0
    best_path = f"best_{model_name}.pth"

    for ep in range(1, epochs + 1):
        tr_loss = trainer.train_one_epoch()
        v_loss, v_acc, v_p, v_r, v_f1 = trainer.eval_metrics()
        print(f"{model_name} | Epoch {ep:02d} | TrainLoss {tr_loss:.4f} | ValLoss {v_loss:.4f} | ValAcc {v_acc*100:.2f}% | ValF1 {v_f1*100:.2f}%")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)

    # Test
    print(f"\n=== {model_name} Test Evaluation ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    trainer = Trainer(model, train_loader, test_loader, optimizer, device, criterion)
    t_loss, t_acc, t_p, t_r, t_f1 = trainer.eval_metrics()
    print(f"{model_name} TEST -> Loss: {t_loss:.4f} | Acc: {t_acc*100:.2f}% | Precision: {t_p*100:.2f}% | Recall: {t_r*100:.2f}% | F1: {t_f1*100:.2f}%")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ds = DatasetLoader(DATA_DIR)

    for model_name in ["SimpleCNN", "ResNet18"]:
        print(f"\n=== Hyperparameter Optimization for {model_name} ===")

        ga = GeneticSearch(
            param_space, ds, device,
            population_size=6,
            generations=5,
            mutation_rate=0.30,
            fitness_epochs=FITNESS_EPOCHS,
            balanced=BALANCED_SAMPLING
        )

        best_ga, _ = ga.search()
        print(f"\nBest GA hyperparams for {model_name}: {best_ga}")

        # Full training
        train_full(model_name, best_ga, epochs=50)
