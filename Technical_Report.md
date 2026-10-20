# Intel Image Classification — Technical Report

## 1. Objective
The objective is to classify natural scene images into six classes:
buildings, forest, glacier, mountain, sea, and street.

Two approaches were evaluated:
- SimpleCNN trained from scratch
- ResNet-18 using transfer learning and fine-tuning

## 2. Dataset and Preparation
- Original training images: 14,034
- Training split: 11,227
- Validation split: 2,807
- Separate test images: 3,000
- Unlabeled prediction images: 7,301
- Number of classes: 6
- Corrupted images found during initial checking: 0

Images were resized to 224 x 224 pixels. Training data used
random horizontal flipping, random rotation, and color jitter.
Images were converted to tensors and normalized.

The test dataset was kept separate from training and validation.

## 3. Models

### SimpleCNN
A custom CNN was built using convolutional layers, ReLU activation,
max pooling, adaptive average pooling, and fully connected layers.
Dropout (0.3) was used in the classifier.

- Trainable parameters: 101,894
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 32
- Training epochs: 10

### ResNet-18 Feature Extraction
Pretrained ResNet-18 weights were used. The pretrained backbone
was frozen and the final classification layer was replaced to
predict six classes.

- Total parameters: 11,179,590
- Trainable parameters: 3,078
- Training epochs: 5

### ResNet-18 Fine-Tuning
The ResNet-18 layer4 block and final classification layer were
unfrozen and trained with a learning rate of 0.0001 for 5 epochs.

- Total parameters: 11,179,590
- Trainable parameters: 8,396,806

## 4. Hyperparameter Experiments
Three controlled experiments were compared with the baseline CNN.

| Experiment | Learning Rate | Batch Size | Dropout | Validation Accuracy |
|---|---:|---:|---:|---:|
| Baseline | 0.001 | 32 | 0.3 | 80.30% |
| Lower learning rate | 0.0005 | 32 | 0.3 | 75.20% |
| Larger batch | 0.001 | 64 | 0.3 | 76.70% |
| Higher dropout | 0.001 | 32 | 0.5 | 75.81% |

The baseline configuration achieved the highest validation accuracy
among these experiments.

## 5. Final Test Results

| Metric | SimpleCNN | Fine-tuned ResNet-18 |
|---|---:|---:|
| Accuracy | 79.93% | 92.63% |
| Macro Precision | 80.20% | 92.84% |
| Macro Recall | 79.95% | 92.81% |
| Macro F1-score | 79.83% | 92.74% |
| Weighted F1-score | 79.76% | 92.63% |
| Correct predictions | 2,398 / 3,000 | 2,779 / 3,000 |

Fine-tuned ResNet-18 improved test accuracy by 12.70 percentage points
compared with the custom CNN.

## 6. Per-Class Results: Fine-Tuned ResNet-18

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Buildings | 87.03% | 95.19% | 90.93% |
| Forest | 99.79% | 98.95% | 99.36% |
| Glacier | 86.40% | 90.78% | 88.54% |
| Mountain | 91.84% | 85.71% | 88.67% |
| Sea | 96.35% | 98.43% | 97.38% |
| Street | 95.65% | 87.82% | 91.57% |

Forest and sea had the strongest results. Mountain and glacier
were relatively more challenging, while buildings and street
still had some classification errors.

## 7. Error Analysis
The fine-tuned ResNet-18 incorrectly classified 221 of the 3,000
test images. The first 10 misclassified examples and their predicted
classes and confidence scores were saved to:
outputs/metrics/resnet18_misclassified_images.csv

The model sometimes confuses visually similar scenes. Confidence
scores should not be interpreted as guarantees of correctness.

## 8. Unseen Image Predictions
A reusable prediction function was created. It returns a predicted
class and confidence scores for all six classes.

Predictions for 10 unlabeled images from seg_pred were saved to:
outputs/metrics/resnet18_unseen_predictions.csv

These images have no provided labels, so their accuracy could not
be measured.

## 9. Generalization and Regularization
Training augmentation (horizontal flips, rotation, and color jitter)
increases variation in training examples. Dropout helps reduce
co-adaptation of neurons in the CNN classifier.

The higher-dropout experiment achieved lower validation accuracy
than the baseline in the tested setup. Fine-tuning showed a gap
between training and validation performance in later epochs,
suggesting possible overfitting.

## 10. Reproducibility and Saved Artifacts
The train-validation split used a fixed random seed. Model weights,
training curves, confusion matrices, and prediction/error-analysis
CSVs were saved in the project directory.

Saved models:
- models/cnn_baseline.pth
- models/resnet18_feature_extraction.pth
- models/resnet18_finetuned.pth

Plots:
- outputs/plots/training_validation_loss.png
- outputs/plots/training_validation_accuracy.png

Confusion matrices:
- outputs/confusion_matrices/cnn_test_confusion_matrix.png
- outputs/confusion_matrices/resnet18_test_confusion_matrix.png

## 11. Limitations and Future Improvements
- The unlabeled prediction set cannot be quantitatively evaluated.
- Confidence scores can be high even for incorrect predictions.
- The saved ResNet checkpoints contain the final training epoch,
  not necessarily the epoch with the highest validation accuracy.
- Future work should save the best checkpoint during training,
  tune hyperparameters further, inspect more errors, and test
  additional augmentation and regularization strategies.

## 12. Conclusion
Fine-tuned ResNet-18 performed substantially better than the
custom CNN on the held-out test set, reaching 92.63% accuracy.
Transfer learning provided a strong starting point for scene
classification. Error analysis and best-checkpoint saving remain
important areas for further improvement.

## Best ResNet-18 Fine-Tuned Checkpoint

The best checkpoint was selected using validation accuracy.

- Best epoch: 3
- Best validation accuracy: 92.95%
- Test accuracy: 92.13%
- Test images evaluated: 3000
- Correct predictions: 2764
- Misclassified images: 236
- Macro precision: 92.47%
- Macro recall: 92.36%
- Macro F1-score: 92.27%
- Weighted F1-score: 92.09%

### Error Analysis

The confusion matrix is saved at:
`outputs/confusion_matrices/resnet18_best_test_confusion_matrix.png`

The misclassified-image list is saved at:
`outputs/metrics/resnet18_best_misclassified_images.csv`

A visualization of ten misclassified test images is saved at:
`outputs/plots/resnet18_best_10_misclassified.png`

The model achieved strong results on forest and sea images. Glacier recall was lower than the other classes, indicating that some actual glacier images were classified as other scene types.

Confidence scores represent the model's predicted probabilities; a high confidence score does not guarantee that a prediction is correct.



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
