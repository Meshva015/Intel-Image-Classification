
from pathlib import Path

import torch
from PIL import Image

from .preprocessing import get_eval_transform


def predict_image(model, image_path, class_names, device, image_size=224):
    """Predict the class and confidence scores for one image."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transform(image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": probabilities[predicted_index].item(),
        "all_confidences": {
            class_names[i]: probabilities[i].item()
            for i in range(len(class_names))
        }
    }
