
from pathlib import Path
from torchvision import datasets


def load_image_dataset(data_dir, transform=None):
    """
    Load an image classification dataset using ImageFolder.

    Expected folder structure:
    data_dir/
        buildings/
        forest/
        glacier/
        mountain/
        sea/
        street/
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    dataset = datasets.ImageFolder(
        root=str(data_dir),
        transform=transform
    )

    return dataset
