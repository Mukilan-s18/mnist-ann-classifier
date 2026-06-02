"""
MNIST Image Classification Using a Simple Artificial Neural Network
====================================================================
A basic fully connected neural network (ANN) to classify handwritten
digit images (0-9) from the MNIST dataset.

Covers:
    - Data loading & preprocessing
    - Model construction with Dense layers
    - Training & validation
    - Evaluation on the test set
    - Visualization of training history and sample predictions
    - Exporting model weights to JSON for web-based inference
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import os
import json

from tensorflow import keras
from tensorflow.keras import layers

def load_data():
    """Loads the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, x_test, y_train, y_test, num_classes=10):
    """Preprocesses and normalizes MNIST images and labels."""
    # Flatten 28x28 images → 784-element vectors
    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype("float32")
    x_test_flat  = x_test.reshape(x_test.shape[0], -1).astype("float32")

    # Normalize pixel values from [0, 255] → [0, 1]
    x_train_flat /= 255.0
    x_test_flat  /= 255.0

    # One-hot encode labels
    y_train_ohe  = keras.utils.to_categorical(y_train, num_classes)
    y_test_ohe   = keras.utils.to_categorical(y_test, num_classes)

    return x_train_flat, x_test_flat, y_train_ohe, y_test_ohe

def build_model(input_shape=(784,), num_classes=10):
    """Builds a Sequential neural network model with three Dense layers."""
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=input_shape, name="hidden_1"),
        layers.Dense(64,  activation="relu",                     name="hidden_2"),
        layers.Dense(num_classes, activation="softmax",          name="output"),
    ])
    return model

def compile_model(model):
    """Compiles the model with optimizer, loss, and metrics."""
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(model, x_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
    """Trains the neural network model."""
    print("\n🚀 Starting training ...\n")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )
    return history

def export_weights_to_json(model, output_path="web/model_weights.json"):
    """Extracts weights and biases from the trained model and exports them to JSON.
    
    This matches the shapes of the Dense layers:
      - Layer 1: W1 (784x128), b1 (128)
      - Layer 2: W2 (128x64), b2 (64)
      - Layer 3: W3 (64x10), b3 (10)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    weights_dict = {
        "W1": model.get_layer("hidden_1").get_weights()[0].tolist(),
        "b1": model.get_layer("hidden_1").get_weights()[1].tolist(),
        "W2": model.get_layer("hidden_2").get_weights()[0].tolist(),
        "b2": model.get_layer("hidden_2").get_weights()[1].tolist(),
        "W3": model.get_layer("output").get_weights()[0].tolist(),
        "b3": model.get_layer("output").get_weights()[1].tolist()
    }
    
    with open(output_path, "w") as f:
        json.dump(weights_dict, f)
    print(f"📦 Model weights successfully exported to JSON → {output_path}")

def save_plots(history, model, x_test_flat, x_test, y_test_ohe, y_test):
    """Saves training history plots and sample predictions."""
    # 1. Training History Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    ax1.plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Loss curves
    ax2.plot(history.history["loss"],     label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("📊 Training history plot saved → training_history.png")

    # 2. Sample Predictions Plot
    predictions = model.predict(x_test_flat[:10], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels    = y_test[:10]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i], cmap="gray")
        color = "green" if predicted_labels[i] == actual_labels[i] else "red"
        ax.set_title(
            f"Pred: {predicted_labels[i]}  |  Actual: {actual_labels[i]}",
            fontsize=10, color=color, fontweight="bold",
        )
        ax.axis("off")

    plt.suptitle("Sample Predictions (green = correct, red = wrong)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    print("🖼️  Sample predictions plot saved → sample_predictions.png")

def main():
    print("=" * 60)
    print("  MNIST Image Classification – Simple ANN")
    print("=" * 60)

    # 1. Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    print("\nDataset loaded successfully!")
    print(f"  Training samples : {x_train.shape[0]}  | Image shape: {x_train.shape[1:]}")
    print(f"  Test samples     : {x_test.shape[0]}   | Image shape: {x_test.shape[1:]}")
    print(f"  Number of classes: {len(np.unique(y_train))}")

    # 2. Preprocess data
    num_classes = 10
    x_train_flat, x_test_flat, y_train_ohe, y_test_ohe = preprocess_data(
        x_train, x_test, y_train, y_test, num_classes
    )
    print("\nAfter preprocessing:")
    print(f"  x_train shape : {x_train_flat.shape}")
    print(f"  x_test  shape : {x_test_flat.shape}")
    print(f"  y_train shape : {y_train_ohe.shape}  (one-hot)")

    # 3. Build & Compile Model
    model = build_model(input_shape=(784,), num_classes=num_classes)
    model.summary()
    model = compile_model(model)

    # 4. Train Model
    EPOCHS = 10
    BATCH_SIZE = 32
    history = train_model(
        model, x_train_flat, y_train_ohe,
        epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2
    )

    # 5. Evaluate Model
    test_loss, test_accuracy = model.evaluate(x_test_flat, y_test_ohe, verbose=0)
    print("\n" + "=" * 60)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
    print("=" * 60)

    # 6. Save visualizations
    save_plots(history, model, x_test_flat, x_test, y_test_ohe, y_test)

    # 7. Export weights for web app
    export_weights_to_json(model, "web/model_weights.json")

    print("\n✅ Done! All steps completed successfully.\n")

if __name__ == "__main__":
    main()

