# Intel Image Classification

## Project Overview
This project classifies natural scene images into six categories
using a CNN built from scratch and a pretrained ResNet-18 model.

## Classes
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Dataset
- Training images: 14,034
- Validation images: 2,807
- Test images: 3,000
- Unseen prediction images: 7,301

The original training dataset was split into training and validation
subsets. The separate test dataset was reserved for final evaluation.

## Models
1. SimpleCNN built from scratch
2. ResNet-18 feature extraction
3. ResNet-18 fine-tuning

## Final Test Results

| Metric | CNN | Fine-tuned ResNet-18 |
|---|---:|---:|
| Accuracy | 79.93% | 92.63% |
| Macro F1-score | 79.83% | 92.74% |
| Weighted F1-score | 79.76% | 92.63% |

## Project Outputs
- models/: saved model weights
- outputs/plots/: training and validation curves
- outputs/confusion_matrices/: test confusion matrices
- outputs/metrics/: error analysis and unseen predictions

## Prediction
The notebook includes a reusable prediction function that returns
the predicted class and confidence scores for all six categories.

The seg_pred images are unlabeled, so their prediction accuracy
cannot be verified from the dataset alone.

## Technologies
- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Reproducibility
The notebook uses a fixed random seed for the train-validation split.
Run the notebook cells in order after configuring the dataset paths
and installing the required libraries.