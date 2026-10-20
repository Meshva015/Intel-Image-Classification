
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


@torch.no_grad()
def evaluate_model(model, loader, device, class_names=None):
    """Evaluate a model and return classification metrics."""

    model.eval()
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))) if class_names else None
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))) if class_names else None,
        target_names=class_names,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report
    }
