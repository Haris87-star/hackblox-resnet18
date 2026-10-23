from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from .dataset import image_files


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_duplicate_groups(directory: Path) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in image_files(directory):
        groups[sha256_file(path)].append(str(path))
    return [paths for paths in groups.values() if len(paths) > 1]