from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CLASS_NAMES, Config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class DatasetAudit:
    labeled_images: int
    unlabeled_images: int
    test_images: int
    class_counts: dict[str, int]
    invalid_images: tuple[str, ...]


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def audit_dataset(config: Config) -> DatasetAudit:
    invalid: list[str] = []
    counts = {name: 0 for name in CLASS_NAMES}
    labeled = labeled_samples(config)
    for path, _ in labeled:
        relative_parts = path.relative_to(config.labeled_dir).parts
        if not relative_parts or relative_parts[0] not in counts:
            invalid.append(str(path))
        else:
            counts[relative_parts[0]] += 1
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid.append(str(path))
    return DatasetAudit(
        labeled_images=len(labeled),
        unlabeled_images=len(image_files(config.unlabeled_dir)),
        test_images=len(image_files(config.test_dir)),
        class_counts=counts,
        invalid_images=tuple(sorted(set(invalid))),
    )


def labeled_samples(config: Config) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = config.labeled_dir / class_name
        samples.extend((path, class_index) for path in image_files(class_dir))
    return sorted(samples, key=lambda item: str(item[0]))