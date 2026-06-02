# MNIST Image Classification Using a Simple ANN

A simple fully-connected Artificial Neural Network (ANN) to classify handwritten digits (0-9) from the classic MNIST dataset. It is implemented using Python, TensorFlow/Keras, NumPy, and Matplotlib.

## Project Structure

- `mnist_ann_classifier.py`: The main script that loads data, preprocesses it, builds and compiles the model, runs training, evaluates performance, and saves visualizations.
- `requirements.txt`: List of Python packages required for the project.
- `training_history.png`: A plot showing training/validation accuracy and loss over epochs.
- `sample_predictions.png`: A visualization of the model's predictions on a sample of test images.

## Model Architecture

The neural network is built with:
- **Input layer**: Flattened 28x28 images to a 784-element vector.
- **Hidden Layer 1**: Dense layer with 128 units and ReLU activation.
- **Hidden Layer 2**: Dense layer with 64 units and ReLU activation.
- **Output Layer**: Dense layer with 10 units (one for each digit 0-9) and Softmax activation.

## Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows (Command Prompt):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the classifier script to train the model, evaluate it, and generate plots:
```bash
python mnist_ann_classifier.py
```

The script will:
1. Load the MNIST dataset (automatically downloaded if not cached locally).
2. Preprocess and normalize pixel values to `[0, 1]`.
3. Train the network for 10 epochs.
4. Print the final test set accuracy (typically >97%).
5. Save `training_history.png` and `sample_predictions.png` to the root folder.
