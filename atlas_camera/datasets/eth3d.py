"""ETH3D dataset discovery on local, externally managed dataset roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas_camera.datasets.colmap import ColmapCamera, ColmapImage, read_colmap_cameras, read_colmap_images


@dataclass(frozen=True, slots=True)
class ETH3DDataset:
    root: Path
    cameras_path: Path
    images_path: Path
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]

    def image_path(self, image: ColmapImage) -> Path:
        candidates = [
            self.root / image.name,
            self.images_path.parent / image.name,
            self.root / Path(image.name).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def iter_images(self) -> list[ColmapImage]:
        return [
            self.images[key]
            for key in sorted(self.images)
        ]


def load_eth3d_dataset(root: str | Path) -> ETH3DDataset:
    dataset_root = Path(root)
    cameras_path, images_path = _find_colmap_text_files(dataset_root)
    return ETH3DDataset(
        root=dataset_root,
        cameras_path=cameras_path,
        images_path=images_path,
        cameras=read_colmap_cameras(cameras_path),
        images=read_colmap_images(images_path),
    )


def _find_colmap_text_files(root: Path) -> tuple[Path, Path]:
    direct = (root / "cameras.txt", root / "images.txt")
    if direct[0].is_file() and direct[1].is_file():
        return direct

    for cameras_path in root.rglob("cameras.txt"):
        images_path = cameras_path.parent / "images.txt"
        if images_path.is_file():
            return cameras_path, images_path

    raise FileNotFoundError(
        f"Could not find COLMAP cameras.txt/images.txt under ETH3D root: {root}"
    )
