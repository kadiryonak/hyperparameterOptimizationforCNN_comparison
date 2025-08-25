
# ──────────────────────────────────────────────────────────────────────────────
# file: trainers/trainer.py
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device: torch.device, criterion: nn.Module | None = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        running = 0.0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(data)
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            running += loss.item() * data.size(0)
        return running / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval_metrics(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_tgts = [], []
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, target)
            total_loss += loss.item() * data.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_tgts.extend(target.cpu().numpy())
        all_preds = np.array(all_preds)
        all_tgts = np.array(all_tgts)
        acc = float((all_preds == all_tgts).mean())
        prec = float(precision_score(all_tgts, all_preds, average='macro', zero_division=0))
        rec = float(recall_score(all_tgts, all_preds, average='macro', zero_division=0))
        f1 = float(f1_score(all_tgts, all_preds, average='macro', zero_division=0))
        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss, acc, prec, rec, f1

    def fit(self, epochs: int = 1, verbose: bool = False):
        best_acc = -1.0
        for ep in range(1, epochs + 1):
            tr_loss = self.train_one_epoch()
            v_loss, v_acc, v_p, v_r, v_f1 = self.eval_metrics()
            if verbose:
                print(f"Epoch {ep:02d} | TrainLoss {tr_loss:.4f} | ValLoss {v_loss:.4f} | ValAcc {v_acc*100:.2f}% | ValF1 {v_f1*100:.2f}%")
            if v_acc > best_acc:
                best_acc = v_acc
        return best_acc
