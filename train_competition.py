"""Rule-compliant scratch ResNet-18 training for the HackBlox scene kit."""

from pathlib import Path
import random
import shutil
import os
import pandas as pd

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
EPOCHS = 35
BATCH_SIZE = 32
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SceneDataset(Dataset):
    def __init__(self, paths: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


class ScratchResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        features = self.resnet.fc.in_features
        setattr(self.resnet, "fc", nn.Identity())
        self.classifier = nn.Sequential(
            nn.Linear(features, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.resnet(images))


def samples_for(split: str, include_pseudo: bool = False) -> list[tuple[Path, int]]:
    result: list[tuple[Path, int]] = []
    for index, name in enumerate(CLASSES):
        result.extend((path, index) for path in sorted((ROOT / "data" / split / name).glob("*.jpg")))
    if split == "train" and include_pseudo:
        pseudo_path = ROOT / "outputs" / os.environ.get("PSEUDO_FILE", "pseudo_labels_120.csv")
        if pseudo_path.exists():
            pseudo = pd.read_csv(pseudo_path)
            result.extend((Path(row.image_path), int(row.pseudo_label_index)) for row in pseudo.itertuples())
    return result


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            flipped = model(torch.flip(images, dims=[3]))
            predictions = (logits.softmax(1) + flipped.softmax(1)).argmax(1)
            correct += int((predictions == labels).sum())
            total += len(labels)
    return correct / max(1, total)


def main() -> None:
    seed_everything(SEED)
    if (ROOT / "best_model.pth").exists():
        shutil.copy2(ROOT / "best_model.pth", ROOT / "outputs" / "best_model_before_final_train.pth")
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomRotation(8), transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32), transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(), normalize,
    ])
    include_pseudo = os.environ.get("USE_PSEUDO", "0") == "1"
    train_samples = samples_for("train", include_pseudo=include_pseudo)
    train_loader = DataLoader(SceneDataset(train_samples, train_transform), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SceneDataset(samples_for("val"), eval_transform), BATCH_SIZE, shuffle=False)
    model = ScratchResNet18().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")
    best = 0.0
    print(f"Training scratch ResNet-18 on {len(train_loader.dataset)} images with {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=DEVICE.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        accuracy = evaluate(model, val_loader)
        print(f"epoch={epoch + 1:02d}/{EPOCHS} val_tta_accuracy={accuracy:.4f}")
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), ROOT / "best_model.pth")
    print(f"best_validation_accuracy={best:.4f}")


if __name__ == "__main__":
    main()