
from torchvision import transforms

# Image size used by the models
IMAGE_SIZE = 224

# Normalization values for ResNet18 pretrained weights
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size=IMAGE_SIZE):
    """Resize images and apply data augmentation during training."""

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_eval_transform(image_size=IMAGE_SIZE):
    """Resize and normalize images without random augmentation."""

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
