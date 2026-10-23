from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CLASS_NAMES, Config
from .dataset import image_files
from .model import build_model


class UnlabeledDataset(Dataset):
    def __init__(self, paths: list[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a schema-checked Kaggle submission.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=None)
    args = parser.parse_args()
    config = Config(root=args.root.resolve()) if args.root else Config()
    checkpoint = config.output_dir / "best_model.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    test_paths = image_files(config.test_dir)
    if not test_paths:
        raise SystemExit("No test images found under data/test/.")
    sample_path = args.sample_submission or config.root / "sample_submission.csv"
    if not sample_path.exists():
        raise SystemExit("sample_submission.csv is required; do not guess the Kaggle schema.")
    sample = pd.read_csv(sample_path)
    if len(sample) != len(test_paths):
        raise SystemExit(f"Sample submission rows ({len(sample)}) do not match test images ({len(test_paths)}).")
    prediction_columns = [column for column in sample.columns if column.lower() == "prediction"]
    if len(prediction_columns) != 1:
        raise SystemExit(f"Could not identify the prediction column in {list(sample.columns)}.")
    prediction_column = prediction_columns[0]
    confidence_columns = [column for column in sample.columns if column.lower() == "confidence"]
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([transforms.Resize(config.image_size + 32), transforms.CenterCrop(config.image_size), transforms.ToTensor(), normalize])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASS_NAMES)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    predictions: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for images, _ in DataLoader(UnlabeledDataset(test_paths, transform), config.batch_size, shuffle=False):
            probabilities = model(images.to(device)).softmax(dim=1)
            confidence, prediction = probabilities.max(1)
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    sample[prediction_column] = predictions
    if confidence_columns:
        sample[confidence_columns[0]] = confidences
    destination = config.output_dir / "submission.csv"
    sample.to_csv(destination, index=False)
    print(f"Wrote {destination} with {len(sample)} predictions in column {prediction_column!r}.")


if __name__ == "__main__":
    main()