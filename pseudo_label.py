"""Create a reversible, confidence-only pseudo-label experiment table."""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import tlc
import os

from train import ResNet18Classifier, CLASS_NAMES, BATCH_SIZE
from register_tables import schemas, PROJECT_NAME, DATASET_NAME


class Images(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), str(path)


def main():
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier().to(device)
    model.load_state_dict(torch.load(root / "best_model.pth", map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(150), transforms.CenterCrop(150), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    paths = sorted((root / "data" / "train" / "undefined").glob("*.jpg"))
    predictions = []
    with torch.no_grad():
        for images, batch_paths in DataLoader(Images(paths, transform), BATCH_SIZE, shuffle=False):
            probabilities = model(images.to(device)).softmax(1)
            confidence, labels = probabilities.max(1)
            predictions.extend((str(path), int(label), float(score)) for path, label, score in zip(batch_paths, labels.cpu(), confidence.cpu()))
    chosen = []
    per_class = int(os.environ.get("PSEUDO_PER_CLASS", "50"))
    for label in range(len(CLASS_NAMES)):
        group = sorted((row for row in predictions if row[1] == label), key=lambda row: row[2], reverse=True)
        chosen.extend(group[:per_class])
    chosen_paths = {path: (label, confidence) for path, label, confidence in chosen}
    table = tlc.Table.from_names(project_name=PROJECT_NAME, dataset_name=DATASET_NAME, table_name="train").latest()
    writer = tlc.TableWriter(
        table_name="train-pseudo-120", dataset_name=DATASET_NAME, project_name=PROJECT_NAME,
        description=f"Reversible confidence-only pseudo-label experiment; {len(chosen)} rows",
        column_schemas=schemas, if_exists="overwrite", input_tables=[table.url],
    )
    for row in table.table_rows:
        image = str(row["image"])
        local_path = str((root / image.replace("<INTEL_SCENE_DATA>/", "")).resolve()) if image.startswith("<INTEL_SCENE_DATA>/") else image
        update = chosen_paths.get(local_path)
        if update is None:
            writer.add_row(dict(row))
        else:
            label, _ = update
            writer.add_row({"id": row["id"], "image": row["image"], "label": label, "weight": 1.0})
    result = writer.finalize()
    output = root / "outputs" / f"pseudo_labels_{len(chosen)}.csv"
    import pandas as pd
    pd.DataFrame(chosen, columns=["image_path", "pseudo_label_index", "confidence"]).to_csv(output, index=False)
    print(f"Created {result.url}")
    print(f"Pseudo-labeled {len(chosen)} rows; confidence range {min(row[2] for row in chosen):.4f}-{max(row[2] for row in chosen):.4f}")
    print(f"Review artifact: {output}")


if __name__ == "__main__":
    main()