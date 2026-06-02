import numpy as np
from tensorflow import keras

# Import the modular functions from the classifier
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mnist_ann_classifier as classifier

def test_load_data():
    """Verify that dataset loading returns training and test sets with correct sizes."""
    (x_train, y_train), (x_test, y_test) = classifier.load_data()
    
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    
    assert len(x_train.shape) == 3  # (num_samples, 28, 28)
    assert x_train.shape[1:] == (28, 28)
    assert x_test.shape[1:] == (28, 28)
    assert len(y_train.shape) == 1  # 1D labels array
    assert len(y_test.shape) == 1

def test_preprocess_data():
    """Verify that preprocessing normalizes data, flattens images, and one-hot encodes labels."""
    # Create mock inputs (simulating pixel values 0-255)
    mock_x_train = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    mock_x_test  = np.random.randint(0, 256, size=(5, 28, 28), dtype=np.uint8)
    mock_y_train = np.random.randint(0, 10, size=(10,), dtype=np.uint8)
    mock_y_test  = np.random.randint(0, 10, size=(5,), dtype=np.uint8)
    
    x_train_flat, x_test_flat, y_train_ohe, y_test_ohe = classifier.preprocess_data(
        mock_x_train, mock_x_test, mock_y_train, mock_y_test, num_classes=10
    )
    
    # 1. Flatten check: (N, 28, 28) -> (N, 784)
    assert x_train_flat.shape == (10, 784)
    assert x_test_flat.shape == (5, 784)
    
    # 2. Normalization check: min/max pixel values inside [0, 1] bounds
    assert np.all(x_train_flat >= 0.0)
    assert np.all(x_train_flat <= 1.0)
    assert np.all(x_test_flat >= 0.0)
    assert np.all(x_test_flat <= 1.0)
    
    # 3. One-hot encoding check: (N,) -> (N, 10)
    assert y_train_ohe.shape == (10, 10)
    assert y_test_ohe.shape == (5, 10)
    
    # Sum of probability distribution for OHE should always be exactly 1
    assert np.allclose(y_train_ohe.sum(axis=1), 1.0)
    assert np.allclose(y_test_ohe.sum(axis=1), 1.0)

def test_build_model():
    """Verify that the built neural network model matches the designated configuration."""
    model = classifier.build_model(input_shape=(784,), num_classes=10)
    
    assert isinstance(model, keras.Sequential)
    assert len(model.layers) == 3
    
    # Check layer configurations
    layer_1 = model.get_layer("hidden_1")
    layer_2 = model.get_layer("hidden_2")
    output_layer = model.get_layer("output")
    
    assert isinstance(layer_1, keras.layers.Dense)
    assert isinstance(layer_2, keras.layers.Dense)
    assert isinstance(output_layer, keras.layers.Dense)
    
    # Check activation functions
    assert layer_1.activation.__name__ == 'relu'
    assert layer_2.activation.__name__ == 'relu'
    assert output_layer.activation.__name__ == 'softmax'
    
    # Check weight dimensions
    # Layer 1: weights (784, 128)
    assert layer_1.weights[0].shape == (784, 128)
    assert layer_1.weights[1].shape == (128,)
    # Layer 2: weights (128, 64)
    assert layer_2.weights[0].shape == (128, 64)
    assert layer_2.weights[1].shape == (64,)
    # Layer 3: weights (64, 10)
    assert output_layer.weights[0].shape == (64, 10)
    assert output_layer.weights[1].shape == (10,)
