# Interactive MNIST Digit Classifier — ANN with Full Evaluation Pipeline

[![CI Pipeline](https://github.com/Mukilan-s18/mnist-ann-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Mukilan-s18/mnist-ann-classifier/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-blueviolet.svg)](https://scikit-learn.org)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://mukilan-s18.github.io/mnist-ann-classifier/)
[![Open in Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](notebook.ipynb)

An **end-to-end, production-style deep learning pipeline** for classifying handwritten digits (0–9) from the MNIST dataset. Features a rigorously evaluated ANN with Dropout regularization, multi-architecture comparison, CNN benchmarking, hyperparameter sensitivity analysis, and a **live interactive browser demo** powered by client-side JavaScript inference.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **Dropout Regularization** | `Dropout(0.2)` after each hidden layer prevents overfitting |
| **Model Save / Load** | Trained model persisted to disk — no retraining on every run |
| **Confusion Matrix** | Full 10×10 heatmap showing where the model makes mistakes |
| **Per-Class Accuracy** | Precision, Recall, F1 for every digit (0–9) via `classification_report` |
| **Architecture Comparison** | 3 ANN variants compared on accuracy and parameter count |
| **CNN Benchmark** | Accuracy vs complexity tradeoff: ANN vs CNN comparison |
| **Hyperparameter Analysis** | Batch size and dropout rate sensitivity sweeps with plots |
| **Early Stopping** | Monitors validation loss, restores best weights automatically |
| **Jupyter Notebook** | Step-by-step interactive narrative in `notebook.ipynb` |
| **Browser Demo** | Draw digits on canvas → real-time predictions via JS inference |
| **CI/CD Pipeline** | GitHub Actions runs `ruff` + `pytest` on every push |

---

## 🧠 Model Architecture

```
Input (784) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(10, Softmax)
```

### Feedforward Math

$$h_1 = \text{ReLU}(W_1 \cdot x + b_1)$$
$$h_2 = \text{ReLU}(W_2 \cdot h_1 + b_2)$$
$$\hat{y} = \text{Softmax}(W_3 \cdot h_2 + b_3)$$

**Total Parameters:** ~109,386 | **Optimizer:** Adam | **Loss:** Categorical Crossentropy

---

## 📊 Results

### Accuracy: **>97.5%** on the MNIST test set (10,000 samples)

### Training Curves
![Training History](training_history.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Sample Predictions
![Sample Predictions](sample_predictions.png)

### Architecture Comparison

| Architecture | Test Accuracy | Parameters |
|---|---|---|
| ANN Small (64→32) | ~97.1% | ~52,000 |
| **ANN Medium (128→64)** ✅ | **~97.7%** | **~109,000** |
| ANN Large (256→128→64) | ~97.8% | ~236,000 |
| CNN (Conv32→Conv64→Dense64) | ~99.1% | ~94,000 |

> **Insight:** The medium ANN offers the best accuracy-to-complexity ratio. The CNN achieves ~1.5% higher accuracy using fewer parameters by exploiting spatial structure.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- `pip`

### 1. Clone & Install

```bash
git clone https://github.com/Mukilan-s18/mnist-ann-classifier.git
cd mnist-ann-classifier
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💻 Usage

### Train the Full Pipeline
```bash
python mnist_ann_classifier.py
```

This will:
1. Train the ANN (or load saved model if it exists)
2. Print test accuracy, loss
3. Generate `confusion_matrix.png`
4. Print per-class classification report
5. Generate `sample_predictions.png`, `training_history.png`
6. Run architecture comparison → `architecture_comparison.png`
7. Run CNN benchmark → `benchmark_comparison.png`
8. Run hyperparameter analysis → `hyperparameter_analysis.png`
9. Export weights → `docs/model_weights.json`

### Run the Interactive Notebook
```bash
jupyter notebook notebook.ipynb
```

### Launch the Browser Demo Locally
```bash
python -m http.server 8000 --directory docs
```
Open **`http://localhost:8000`** — draw any digit and watch the model predict in real-time.

### Run Tests
```bash
pytest tests/ -v
```

### Code Quality Check
```bash
ruff check .
```

---

## 📁 Project Structure

```
mnist-ann-classifier/
├── mnist_ann_classifier.py    # Full ML pipeline (train, eval, compare, benchmark)
├── notebook.ipynb             # Step-by-step Jupyter walkthrough
├── requirements.txt           # Pinned dependencies
├── training_history.png       # Training/validation accuracy & loss curves
├── confusion_matrix.png       # 10x10 confusion matrix heatmap
├── sample_predictions.png     # Grid of test predictions with labels
├── docs/                      # GitHub Pages web demo
│   ├── index.html             # Interactive canvas UI
│   ├── style.css              # Dark glassmorphism theme
│   ├── script.js              # Client-side neural network inference
│   └── model_weights.json     # Exported weights for browser inference
├── tests/
│   └── test_classifier.py     # pytest suite (data, model, save/load)
└── .github/
    └── workflows/ci.yml       # GitHub Actions CI pipeline
```

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
