# Interactive MNIST Digit Classifier Using a Simple ANN

[![CI Pipeline](https://github.com/mukilan/mnist-ann-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/mukilan/mnist-ann-classifier/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://mukilan-s18.github.io/mnist-ann-classifier/)

An end-to-end, highly visual project implementing a fully-connected Artificial Neural Network (ANN) to classify handwritten digits (0-9) from the classic MNIST dataset. It features a Python training pipeline built with TensorFlow/Keras and an **interactive, client-side web application** that runs the trained neural network model in real-time inside your browser using pure, vanilla JavaScript.

---

## 🚀 Key Features

* **Interactive Web Canvas**: Draw any digit (0-9) on the HTML5 canvas and see predictions update instantly.
* **Zero-Dependency JS Inference**: Trained weights are exported to JSON (`web/model_weights.json`) and computed on the fly in the browser using custom matrix algebra—no heavy libraries (like TensorFlow.js or ONNX) required!
* **Mobile Friendly**: Canvas drawing supports both mouse drag and touch events for phones and tablets.
* **Modular Pipeline**: Clean, structured Python code (`mnist_ann_classifier.py`) divided into testable preprocessing, compilation, training, and export functions.
* **Robust Verification**: Unit tests (`pytest`) verify data pipelines and model constraints; linting (`ruff`) enforces clean code styles.
* **CI/CD Integration**: Automated GitHub Actions workflow checks syntax and runs the test suite on every push.

---

## 🎨 Interactive Web Demo

The `web/` folder is designed for quick deployment via **GitHub Pages**. Once pushed, the live demo is accessible at:
👉 **`https://<your-username>.github.io/mnist-ann-classifier/`**

### Preview
Below is the web application's dashboard running client-side predictions:

* **Subsampled Input**: The drawing area is automatically downscaled to a $28 \times 28$ grayscale grid, matching the exact format the network was trained on.
* **Class Probabilities**: A dynamic bar chart displays activation probabilities calculated from the output Softmax layer.

---

## 🧠 Model Architecture & Mathematics

The network is a fully-connected Feedforward Neural Network (Multi-Layer Perceptron):

```
[Input Layer]       [Hidden Layer 1]       [Hidden Layer 2]       [Output Layer]
 784 Neurons   ──>    128 Neurons    ──>     64 Neurons    ──>     10 Neurons
(28x28 Image)            (ReLU)                 (ReLU)              (Softmax)
```

### Feedforward Propagation Math

Every prediction is calculated using standard linear algebra:
1. **Hidden Layer 1**:  
   $$h_1 = \max(0, W_1 \cdot x + b_1)$$
2. **Hidden Layer 2**:  
   $$h_2 = \max(0, W_2 \cdot h_1 + b_2)$$
3. **Output Layer (Softmax)**:  
   $$y = \text{Softmax}(W_3 \cdot h_2 + b_3)$$

Where:
* $x$ is the $784$-dimensional normalized input vector ($[0.0, 1.0]$ grayscale values).
* $W_1$, $W_2$, $W_3$ are the weight matrices.
* $b_1$, $b_2$, $b_3$ are the bias vectors.
* $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{k=0}^{9} e^{z_k}}$ yields class probabilities.

---

## 📊 Training Results

Running the training script compiles the Keras model using the **Adam** optimizer and **Categorical Crossentropy** loss. It trains for 10 epochs (typically achieving **>97.5% validation accuracy**):

### 1. Training Curves (`training_history.png`)
![Training History](training_history.png)

### 2. Test Set Predictions (`sample_predictions.png`)
![Sample Predictions](sample_predictions.png)

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9 or higher
* `pip` (Python package manager)

### Local Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mukilan/mnist-ann-classifier.git
   cd mnist-ann-classifier
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate.bat
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage Instructions

### 1. Train and Export Weights
To run the training pipeline, evaluate the model on the test set, save training visualizations, and export weights:
```bash
python mnist_ann_classifier.py
```
This writes the neural network weights to `web/model_weights.json`.

### 2. Launch the Web Interface Locally
Since the browser fetches `model_weights.json` asynchronously, loading the HTML directly via `file://` is blocked by browser CORS security policies. Run a local web server to play with the demo:
```bash
python -m http.server 8000 --directory web
```
Open **`http://localhost:8000`** in your browser.

### 3. Run Verification Tests
To run unit tests validating model structures and data formatting:
```bash
pytest tests/
```

To verify code formatting and styling:
```bash
ruff check .
```

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome! Please review the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines and adhere to the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
