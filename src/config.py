from dataclasses import dataclass, field
from pathlib import Path


CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")


@dataclass
class Config:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 0
    patience: int = 7
    max_training_samples: int = 3000

    @property
    def labeled_dir(self) -> Path:
        return self.root / "data" / "train"

    @property
    def unlabeled_dir(self) -> Path:
        return self.root / "data" / "train" / "undefined"

    @property
    def test_dir(self) -> Path:
        return self.root / "data" / "test"

    @property
    def validation_dir(self) -> Path:
        return self.root / "data" / "val"

    @property
    def output_dir(self) -> Path:
        return self.root / "outputs"