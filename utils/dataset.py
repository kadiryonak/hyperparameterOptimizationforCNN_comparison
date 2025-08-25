import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter


class DatasetLoader:
    def __init__(self, data_dir: str, seed: int = 42, img_size=(28, 28)):
        self.data_dir = data_dir
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.class_to_idx = self.full_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        total_len = len(self.full_dataset)
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len

        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.full_dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def _make_sampler(self, dataset):
        labels = [y for _, y in dataset]
        counts = Counter(labels)
        total = sum(counts.values())
        class_weights = {c: total / (len(counts) * cnt) for c, cnt in counts.items()}
        weights = [class_weights[y] for y in labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def make_loaders(self, batch_size: int, balanced: bool = False, device: torch.device | None = None):
        pin = (device is not None and device.type == "cuda")
        if balanced:
            sampler = self._make_sampler(self.train_ds)
            train_loader = DataLoader(self.train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=pin)
        else:
            train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)

        val_loader = DataLoader(self.val_ds, batch_size=max(256, batch_size), shuffle=False, num_workers=0, pin_memory=pin)
        test_loader = DataLoader(self.test_ds, batch_size=max(256, batch_size), shuffle=False, num_workers=0, pin_memory=pin)
        return train_loader, val_loader, test_loader


