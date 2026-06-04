---
license: mit
title: MNIST ANN Digit Classifier
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: true
tags:
  - image-classification
  - mnist
  - deep-learning
  - ann
  - tensorflow
  - keras
  - explainability
  - saliency-maps
---

# Model Card — MNIST ANN Digit Classifier

## Model Overview

| Property | Value |
|---|---|
| **Task** | Multi-class image classification (10 classes) |
| **Architecture** | Fully Connected ANN (3 Dense layers + Dropout) |
| **Framework** | TensorFlow 2.21 / Keras |
| **Input** | Flattened 28×28 greyscale image (784 floats, normalised to [0,1]) |
| **Output** | Softmax probability distribution over digits 0–9 |
| **Parameters** | 109,386 trainable |
| **Author** | Mukilan S |
| **License** | MIT |

---

## Architecture

```
Input (784)
    ↓
Dense(128, activation=ReLU)
    ↓
Dropout(rate=0.2)
    ↓
Dense(64, activation=ReLU)
    ↓
Dropout(rate=0.2)
    ↓
Dense(10, activation=Softmax)   ← Output: class probabilities
```

### Why Dropout?
Dropout randomly deactivates 20% of neurons during each training step. This prevents co-adaptation of neurons and acts as an implicit ensemble — forcing the network to learn robust, redundant representations.

---

## Training Details

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate, fast convergence |
| Loss | Categorical Crossentropy | Standard for multi-class classification |
| Epochs | 15 (with Early Stopping, patience=3) | Prevents overfitting |
| Batch Size | 64 | Best accuracy in hyperparameter sweep |
| Dropout Rate | 0.2 | Optimal in sensitivity analysis |
| Validation Split | 10% | Standard hold-out for monitoring |

---

## Training Data

| Property | Details |
|---|---|
| **Dataset** | MNIST (Modified National Institute of Standards and Technology) |
| **Source** | `tensorflow.keras.datasets.mnist` |
| **Train size** | 60,000 images |
| **Test size** | 10,000 images |
| **Image size** | 28 × 28 pixels, greyscale |
| **Classes** | 10 (digits 0–9), approximately balanced |
| **Normalisation** | Pixel values divided by 255.0 → [0.0, 1.0] |

---

## Evaluation Results

### Overall Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.86%** |
| **Test Loss** | 0.0758 |

### Per-Class Performance

| Digit | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.97 | 0.99 | 0.98 | 980 |
| 1 | 0.99 | 0.99 | 0.99 | 1135 |
| 2 | 0.98 | 0.98 | 0.98 | 1032 |
| 3 | 0.98 | 0.98 | 0.98 | 1010 |
| 4 | 0.98 | 0.98 | 0.98 | 982 |
| 5 | 0.97 | 0.98 | 0.98 | 892 |
| 6 | 0.99 | 0.98 | 0.98 | 958 |
| 7 | 0.96 | 0.98 | 0.97 | 1028 |
| 8 | 0.98 | 0.96 | 0.97 | 974 |
| 9 | 0.99 | 0.97 | 0.98 | 1009 |
| **Macro Avg** | **0.98** | **0.98** | **0.98** | **10000** |

### Architecture Comparison

| Model | Accuracy | Parameters |
|---|---|---|
| ANN Small (64→32) | 97.06% | 52,650 |
| **ANN Medium (128→64) ← this model** | **98.08%** | **109,386** |
| ANN Large (256→128→64) | 97.78% | 242,762 |
| CNN (Conv32→Conv64→Dense64) | 99.07% | ~94,000 |

> The medium ANN offers the best accuracy-to-parameter ratio. A CNN outperforms it by ~1% by exploiting spatial structure — an expected and documented limitation of fully connected networks for image tasks.

---

## Intended Use

✅ **Suitable for:**
- Educational demonstrations of ANN classification
- Prototyping and learning deep learning pipelines
- Benchmarking simple neural network architectures

❌ **Not suitable for:**
- Production handwriting recognition systems
- Non-digit character classification (letters, symbols)
- Real-world cheque/form processing without fine-tuning
- Security-sensitive digit verification

---

## Known Limitations & Failure Cases

| Limitation | Details |
|---|---|
| **Non-centred digits** | Model degrades on digits not centred in the 28×28 frame |
| **Unusual stroke styles** | Very thin or stylised handwriting reduces accuracy |
| **Rotated digits** | No rotation invariance; 6 and 9 may be confused if rotated |
| **Noise sensitivity** | Background noise degrades performance (MNIST has clean backgrounds) |
| **Not CNN-based** | Cannot exploit spatial relationships; CNN achieves ~1% higher accuracy |

---

## Bias & Fairness

The MNIST dataset is well-balanced across all 10 digit classes (9,000–11,000 samples per class in the full dataset). However:
- It reflects handwriting styles predominantly from US Census Bureau employees and American high school students
- May not generalise perfectly to other handwriting cultures or styles
- Digit "1" achieves the highest F1 (0.99); digit "7" the lowest (0.97) — likely due to visual similarity with "1"

---

## Explainability

This model ships with **vanilla gradient saliency maps** — visualisations showing which input pixels most influenced a given prediction. Available in the live demo and via `compute_saliency()` in `app.py`.

---

## How to Reproduce

```bash
git clone https://github.com/Mukilan-s18/mnist-ann-classifier
cd mnist-ann-classifier
make install
make train       # regenerates saved_model/mnist_ann.keras
make demo        # launches Gradio demo at localhost:7860
```

---

## Citation

```bibtex
@misc{mukilan2024mnist,
  author       = {Mukilan S},
  title        = {MNIST ANN Digit Classifier with Full Evaluation Pipeline},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/Mukilan-s18/mnist-ann-classifier}}
}
```
