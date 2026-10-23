"""Rank undefined images for the first human-labeling batch.

This is a local review aid. It never assigns labels or changes the 3LC tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from train import ResNet18Classifier, CLASS_NAMES, BATCH_SIZE


class UndefinedDataset(Dataset):
    def __init__(self, paths: list[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), str(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = sorted((root / "data" / "train" / "undefined").glob("*.jpg"))
    if not paths:
        raise SystemExit("No undefined images found under data/train/undefined.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier(num_classes=6).to(device)
    model.load_state_dict(torch.load(root / "best_model.pth", map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in DataLoader(UndefinedDataset(paths, transform), BATCH_SIZE, shuffle=False):
            features = model.resnet(images.to(device))
            probabilities.append(model.classifier(features).softmax(1).cpu().numpy())
            embeddings.append(features.cpu().numpy())
    probabilities_array = np.concatenate(probabilities)
    embeddings_array = np.concatenate(embeddings)
    ordered = np.sort(probabilities_array, axis=1)
    confidence = ordered[:, -1]
    margin = confidence - ordered[:, -2]
    entropy = -(probabilities_array * np.log(np.clip(probabilities_array, 1e-12, 1.0))).sum(1)
    uncertainty = (entropy - entropy.min()) / max(1e-12, entropy.max() - entropy.min())
    kmeans = MiniBatchKMeans(n_clusters=min(100, len(paths)), random_state=42, batch_size=512, n_init=3)
    cluster = kmeans.fit_predict(embeddings_array)
    cluster_counts = np.bincount(cluster)
    diversity = 1.0 / cluster_counts[cluster]
    diversity = (diversity - diversity.min()) / max(1e-12, diversity.max() - diversity.min())
    score = 0.7 * uncertainty + 0.2 * (1.0 - margin) + 0.1 * diversity
    result = pd.DataFrame({
        "image_path": [str(path) for path in paths],
        "predicted_class": [CLASS_NAMES[index] for index in probabilities_array.argmax(1)],
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "cluster_id": cluster,
        "active_learning_score": score,
    }).sort_values("active_learning_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["reason_for_selection"] = "high uncertainty plus embedding diversity; verify visually"
    destination = root / "outputs" / "undefined_candidates.csv"
    destination.parent.mkdir(exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Wrote {len(result)} ranked undefined images to {destination}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()