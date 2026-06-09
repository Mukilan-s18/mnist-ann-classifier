"""
Unit tests for the MNIST ANN Classifier pipeline.
Covers: data loading, preprocessing, model architecture,
        save/load round-trip, and confusion matrix shape.
"""

import os
import tempfile

import numpy as np
from tensorflow import keras

import mnist_ann_classifier as clf


# ──────────────────────────────────────────────────────────────
# Data tests
# ──────────────────────────────────────────────────────────────
def test_load_data():
    """Dataset should return numpy arrays with expected shapes."""
    (x_train, y_train), (x_test, y_test) = clf.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert len(y_train) == 60000
    assert len(y_test) == 10000


def test_preprocess_data():
    """Preprocessing should flatten, normalize, and one-hot encode."""
    mock_x_train = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
    mock_x_test = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
    mock_y_train = np.random.randint(0, 10, 20, dtype=np.uint8)
    mock_y_test = np.random.randint(0, 10, 10, dtype=np.uint8)

    xtr, xte, ytr, yte = clf.preprocess_data(mock_x_train, mock_x_test, mock_y_train, mock_y_test)

    # Shape checks
    assert xtr.shape == (20, 784)
    assert xte.shape == (10, 784)
    assert ytr.shape == (20, 10)
    assert yte.shape == (10, 10)

    # Normalization bounds
    assert xtr.min() >= 0.0 and xtr.max() <= 1.0
    assert xte.min() >= 0.0 and xte.max() <= 1.0

    # One-hot sums to 1
    assert np.allclose(ytr.sum(axis=1), 1.0)
    assert np.allclose(yte.sum(axis=1), 1.0)


# ──────────────────────────────────────────────────────────────
# Model architecture tests
# ──────────────────────────────────────────────────────────────
def test_build_model_default():
    """Default model should have 3 Dense layers and correct output size."""
    model = clf.build_model()
    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    assert len(dense_layers) == 3
    assert dense_layers[-1].units == 10
    assert dense_layers[-1].activation.__name__ == "softmax"


def test_build_model_custom_units():
    """Model should adapt to custom architecture specification."""
    model = clf.build_model(units=(256, 128, 64))
    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    assert dense_layers[0].units == 256
    assert dense_layers[1].units == 128
    assert dense_layers[2].units == 64
    assert dense_layers[3].units == 10


def test_build_model_dropout():
    """Model should contain Dropout layers equal to number of hidden units."""
    model = clf.build_model(units=(128, 64), dropout_rate=0.3)
    dropout_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dropout)]
    assert len(dropout_layers) == 2  # one per hidden layer


def test_model_output_shape():
    """Model should produce (batch, 10) softmax output."""
    model = clf.build_model()
    dummy_input = np.random.rand(5, 784).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (5, 10)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


# ──────────────────────────────────────────────────────────────
# Save / load round-trip
# ──────────────────────────────────────────────────────────────
def test_save_and_load_model():
    """Saved model should produce identical predictions after loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.keras")
        model = clf.build_model()
        clf.save_model(model, path=save_path)
        assert os.path.exists(save_path)

        loaded = clf.load_saved_model(path=save_path)
        assert loaded is not None

        dummy = np.random.rand(3, 784).astype("float32")
        preds_orig = model.predict(dummy, verbose=0)
        preds_loaded = loaded.predict(dummy, verbose=0)
        assert np.allclose(preds_orig, preds_loaded, atol=1e-5)


def test_load_nonexistent_model():
    """load_saved_model should return None for missing path."""
    result = clf.load_saved_model(path="/tmp/definitely_does_not_exist.keras")
    assert result is None


# ──────────────────────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────────────────────
def test_confusion_matrix_shape():
    """Confusion matrix output should be (10, 10) for MNIST."""
    from sklearn.metrics import confusion_matrix

    y_true = np.random.randint(0, 10, 100)
    y_pred = np.random.randint(0, 10, 100)
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (10, 10)
