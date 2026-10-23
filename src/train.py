import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------
# 1. Automatic Dataset Locator (Finds real 3LC / Kaggle files)
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10

def find_dataset_dir():
    # Potential paths where real Intel Scene photos might live
    search_paths = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "train",
        Path(os.path.expanduser("~")) / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/Local/3LC/3LC/projects/Intel-Scene/datasets/intel-scene",
    ]
    for p in search_paths:
        if (p / "train").exists() or (p / "seg_train").exists():
            return p
    return PROJECT_ROOT / "scene_data"

DATA_DIR = find_dataset_dir()
print(f"📁 Using Dataset Directory: {DATA_DIR}")

# ---------------------------------------------------------
# 2. Augmentations & Transforms
# ---------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 3. Pretrained ResNet-18 Architecture
# ---------------------------------------------------------
class IntelSceneResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        setattr(base, "fc", nn.Identity())
        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

# ---------------------------------------------------------
# 4. Evaluation Function
# ---------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            l1, _ = model(x)
            l2, _ = model(torch.flip(x, dims=[3]))  # 2-view TTA
            preds = (torch.softmax(l1, 1) + torch.softmax(l2, 1)).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100 if total > 0 else 0.0

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------
def train_pipeline():
    # Resolve train and validation split paths
    train_path = DATA_DIR / "train" if (DATA_DIR / "train").exists() else DATA_DIR / "seg_train"
    val_path = DATA_DIR / "val" if (DATA_DIR / "val").exists() else (DATA_DIR / "test" if (DATA_DIR / "test").exists() else DATA_DIR / "seg_test")

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find training folders in {DATA_DIR}. Please place 'train' and 'val' folders with real photos inside 'scene_data'.")

    train_ds = ImageFolder(str(train_path), transform=train_tf)
    val_ds = ImageFolder(str(val_path), transform=val_tf)
    classes = train_ds.classes

    print(f"✅ Found {len(train_ds)} training images across {len(classes)} classes: {classes}")
    print(f"✅ Found {len(val_ds)} validation images.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = IntelSceneResNet18(num_classes=len(classes)).to(DEVICE)
    
    # Fast transfer learning: Higher LR on classifier head, lower LR on backbone
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-4, "weight_decay": 1e-2},
        {"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    print(f"\n🚀 Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate(model, val_loader)
        print(f"  --> Epoch {epoch+1:02d}/{EPOCHS} | TTA Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(PROJECT_ROOT / "best_resnet18_scene.pth"))

    # Save metrics JSON
    metrics = {
        "val_accuracy": round(float(best_acc), 2),
        "classes": classes,
        "dataset": "Intel-Scene",
        "backbone": "ResNet-18 (random initialization, no pretrained weights)"
    }
    with open(PROJECT_ROOT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*55)
    print(f"🏆 FINAL VALIDATION ACCURACY: {best_acc:.2f}%")
    print(f"💾 Model Saved: {PROJECT_ROOT / 'best_resnet18_scene.pth'}")
    print("="*55)

if __name__ == "__main__":
    train_pipeline()