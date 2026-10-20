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

## Best ResNet-18 Checkpoint Results

The best fine-tuned checkpoint was selected using validation accuracy.

| Metric | Result |
|---|---:|
| Best epoch | 3 |
| Validation accuracy | 92.95% |
| Test accuracy | 92.13% |
| Macro precision | 92.47% |
| Macro recall | 92.36% |
| Macro F1-score | 92.27% |
| Weighted F1-score | 92.09% |

### Best Model Files

- Checkpoint: `models/resnet18_finetuned_best.pth`
- Confusion matrix: `outputs/confusion_matrices/resnet18_best_test_confusion_matrix.png`
- Misclassified images CSV: `outputs/metrics/resnet18_best_misclassified_images.csv`
- Ten-image error analysis: `outputs/plots/resnet18_best_10_misclassified.png`

### Error Analysis

The model performs particularly well on forest and sea images. Glacier recall is comparatively lower, so some glacier images are classified as other scene categories.

A high prediction confidence does not guarantee a correct prediction.



## Final Model Comparison

| Model | Best Validation Accuracy | Test Accuracy | Test Loss |
|---|---:|---:|---:|
| SimpleCNN Baseline | 80.30% | 79.93% | 0.5591 |
| ResNet-18 Feature Extraction | 90.59% | 90.00% | 0.2769 |
| ResNet-18 Fine-Tuned (Best Checkpoint) | 92.95% | 92.13% | 0.2288 |

### Conclusion
The fine-tuned ResNet-18 achieved the best test accuracy (92.13%)
and the lowest test loss (0.2288) among the three evaluated models.
The SimpleCNN baseline achieved 79.93% test accuracy.
The fine-tuned model improved test accuracy by approximately
12.20 percentage points over the baseline.

The best fine-tuned checkpoint was selected using validation
performance. The test set was used for final evaluation.



## Environment and Reproducibility

### Software Versions
- Python: 3.14.5
- PyTorch: 2.14.1+cu130
- Torchvision: 0.29.1+cu130
- NumPy: 2.5.2
- Pandas: 3.0.5
- Scikit-learn: 1.9.1
- Matplotlib: 3.11.1

### Hardware
- GPU: NVIDIA GeForce RTX 3050 A Laptop GPU
- CUDA available: Yes

### Reproducibility Notes
- Use the same dataset structure and preprocessing pipeline.
- Use the saved best model checkpoint for final predictions.
- Select models using validation performance and reserve the test set for final evaluation.


### Random Seed
- Seed value: 42
- Python random, NumPy, and PyTorch seeds are set to 42.
- CUDA random seed is also set when CUDA is available.
- For reproducible training, set these seeds before data splitting and model training.

## How to Run

1. Clone or download this project.
2. Install the required Python packages listed in `requirements.txt`.
3. Open `notebooks/exploration.ipynb` in Jupyter Notebook or VS Code.
4. Update the dataset paths if needed.
5. Run the notebook cells in order.

## Dataset Folder Structure

The dataset should contain these folders:

- `archive/seg_train/seg_train/`
- `archive/seg_test/seg_test/`
- `archive/seg_pred/seg_pred/`

The training and test folders should contain one subfolder for each of the six scene classes.