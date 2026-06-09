"""
MNIST Image Classification Using a Simple Artificial Neural Network
====================================================================
A fully-featured ML pipeline to classify handwritten digit images (0-9)
from the MNIST dataset. This script demonstrates:

    - Data loading & preprocessing
    - Model construction with Dense layers + Dropout regularization
    - Save & load trained model (no retraining needed)
    - Training & validation with learning curves
    - Full evaluation: accuracy, confusion matrix, per-class report
    - Architecture comparison across 3 ANN variants
    - CNN benchmark comparison
    - Hyperparameter (epochs) sensitivity analysis
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = "saved_model/mnist_ann.keras"
WEIGHTS_JSON = "docs/model_weights.json"
EPOCHS = 15
BATCH_SIZE = 64
NUM_CLASSES = 10
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────
# 1. Data
# ──────────────────────────────────────────────────────────────
def load_data():
    """Loads the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, x_test, y_train, y_test, num_classes=NUM_CLASSES):
    """Flattens, normalizes images and one-hot encodes labels."""
    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0
    y_train_ohe = keras.utils.to_categorical(y_train, num_classes)
    y_test_ohe = keras.utils.to_categorical(y_test, num_classes)
    return x_train_flat, x_test_flat, y_train_ohe, y_test_ohe


# ──────────────────────────────────────────────────────────────
# 2. Model Builders
# ──────────────────────────────────────────────────────────────
def build_model(input_shape=(784,), num_classes=NUM_CLASSES, units=(128, 64), dropout_rate=0.2):
    """
    Builds a Sequential ANN with Dropout regularization.

    Args:
        input_shape  : shape of flattened input (784,)
        num_classes  : number of output classes (10)
        units        : tuple of hidden layer sizes
        dropout_rate : fraction of neurons dropped during training
    """
    model_layers = [keras.Input(shape=input_shape)]
    for u in units:
        model_layers.append(layers.Dense(u, activation="relu"))
        model_layers.append(layers.Dropout(dropout_rate))
    model_layers.append(layers.Dense(num_classes, activation="softmax", name="output"))

    model = keras.Sequential(model_layers)
    return model


def build_cnn(input_shape=(28, 28, 1), num_classes=NUM_CLASSES):
    """Builds a small CNN for benchmarking against the ANN."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )
    return model


# ──────────────────────────────────────────────────────────────
# 3. Compile & Train
# ──────────────────────────────────────────────────────────────
def compile_model(model):
    """Compiles with Adam optimizer and categorical crossentropy loss."""
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model, x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=2
):
    """
    Trains the neural network.
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return history


# ──────────────────────────────────────────────────────────────
# 4. Save / Load
# ──────────────────────────────────────────────────────────────
def save_model(model, path=MODEL_SAVE_PATH):
    """Saves the trained Keras model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"💾 Model saved → {path}")


def load_saved_model(path=MODEL_SAVE_PATH):
    """Loads a previously saved Keras model from disk."""
    if os.path.exists(path):
        model = keras.models.load_model(path)
        print(f"✅ Loaded saved model from {path}")
        return model
    return None


def export_weights_to_json(model, output_path=WEIGHTS_JSON):
    """Exports weights of the three Dense layers to JSON for browser inference."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
    named = {
        "hidden_1": dense_layers[0],
        "hidden_2": dense_layers[1],
        "output": dense_layers[2],
    }

    weights_dict = {}
    for key, layer in named.items():
        W, b = layer.get_weights()
        weights_dict[f"W_{key}"] = W.tolist()
        weights_dict[f"b_{key}"] = b.tolist()

    # Also keep legacy keys for the web app
    W1, b1 = dense_layers[0].get_weights()
    W2, b2 = dense_layers[1].get_weights()
    W3, b3 = dense_layers[2].get_weights()
    weights_dict.update(
        {
            "W1": W1.tolist(),
            "b1": b1.tolist(),
            "W2": W2.tolist(),
            "b2": b2.tolist(),
            "W3": W3.tolist(),
            "b3": b3.tolist(),
        }
    )

    with open(output_path, "w") as f:
        json.dump(weights_dict, f)
    print(f"📦 Weights exported → {output_path}")


# ──────────────────────────────────────────────────────────────
# 5. Evaluation & Visualization
# ──────────────────────────────────────────────────────────────
def plot_training_history(
    history, title="ANN Training History", save_path="figures/training_history.png"
):
    """Plots accuracy and loss curves from training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val", linewidth=2, linestyle="--")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"], label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val", linewidth=2, linestyle="--")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Training history → {save_path}")


def plot_confusion_matrix(model, x_test, y_test_raw, save_path="figures/confusion_matrix.png"):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    """
    Generates and saves a confusion matrix heatmap.

    Args:
        y_test_raw : integer labels (not one-hot encoded)
    """
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test_raw, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix – MNIST ANN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix → {save_path}")
    return cm


def print_classification_report(model, x_test, y_test_raw):
    from sklearn.metrics import classification_report

    """Prints a per-class precision, recall, and F1-score report."""
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(
        y_test_raw, y_pred, target_names=[f"Digit {i}" for i in range(10)]
    )
    print("\n" + "=" * 60)
    print("  Per-Class Classification Report")
    print("=" * 60)
    print(report)
    return report


def plot_sample_predictions(
    model, x_test, x_test_raw, y_test_raw, n=10, save_path="figures/sample_predictions.png"
):
    """Visualises n test images with predicted vs actual labels."""
    preds = np.argmax(model.predict(x_test[:n], verbose=0), axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(
        "Sample Predictions (green = correct, red = wrong)", fontsize=13, fontweight="bold", y=1.02
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test_raw[i], cmap="gray")
        color = "green" if preds[i] == y_test_raw[i] else "red"
        ax.set_title(
            f"Pred: {preds[i]}  Actual: {y_test_raw[i]}", fontsize=9, color=color, fontweight="bold"
        )
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼️  Sample predictions → {save_path}")


# ──────────────────────────────────────────────────────────────
# 6. Architecture Comparison
# ──────────────────────────────────────────────────────────────
ARCHITECTURES = {
    "ANN-Small  (64→32)": (64, 32),
    "ANN-Medium (128→64)": (128, 64),
    "ANN-Large  (256→128→64)": (256, 128, 64),
}


def compare_architectures(
    x_train, y_train_ohe, x_test, y_test_ohe, save_path="figures/architecture_comparison.png"
):
    """
    Trains 3 ANN variants, compares their validation accuracy,
    and saves a bar chart.
    """
    print("\n" + "=" * 60)
    print("  Architecture Comparison")
    print("=" * 60)

    results = {}
    for name, units in ARCHITECTURES.items():
        print(f"\n▶  Training {name} …")
        m = build_model(units=units)
        m = compile_model(m)
        _ = train_model(m, x_train, y_train_ohe, epochs=10, verbose=0)
        _, acc = m.evaluate(x_test, y_test_ohe, verbose=0)
        params = m.count_params()
        results[name] = {"accuracy": acc * 100, "params": params}
        print(f"   Accuracy: {acc * 100:.2f}%  |  Parameters: {params:,}")

    # Bar chart
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    params = [results[n]["params"] for n in names]
    colors = ["#3b82f6", "#10b981", "#f59e0b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ANN Architecture Comparison", fontsize=14, fontweight="bold")

    bars = ax1.bar(names, accs, color=colors, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_ylim(95, 100)
    ax1.set_title("Accuracy by Architecture")
    for bar, acc in zip(bars, accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{acc:.2f}%",
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
    ax1.tick_params(axis="x", labelrotation=15)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(names, params, color=colors, edgecolor="white", linewidth=1.2)
    ax2.set_ylabel("Parameter Count")
    ax2.set_title("Model Complexity")
    for bar, p in zip(bars2, params):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{p:,}",
            ha="center",
            fontsize=9,
        )
    ax2.tick_params(axis="x", labelrotation=15)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Architecture comparison → {save_path}")
    return results


# ──────────────────────────────────────────────────────────────
# 7. CNN Benchmark
# ──────────────────────────────────────────────────────────────
def benchmark_cnn(
    x_train_raw,
    y_train_ohe,
    x_test_raw,
    y_test_ohe,
    ann_accuracy,
    save_path="figures/benchmark_comparison.png",
):
    """
    Trains a small CNN and compares accuracy vs the best ANN.
    """
    print("\n" + "=" * 60)
    print("  CNN Benchmark")
    print("=" * 60)

    # Reshape for CNN input: (N, 28, 28, 1)
    x_train_cnn = x_train_raw.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test_cnn = x_test_raw.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    cnn = build_cnn()
    cnn = compile_model(cnn)
    print("\n▶  Training CNN …")
    train_model(cnn, x_train_cnn, y_train_ohe, epochs=10, verbose=0)
    _, cnn_acc = cnn.evaluate(x_test_cnn, y_test_ohe, verbose=0)
    print(f"   CNN Accuracy  : {cnn_acc * 100:.2f}%")
    print(f"   Best ANN Acc  : {ann_accuracy:.2f}%")

    # Comparison bar chart
    labels = ["Best ANN\n(256→128→64)", "CNN\n(Conv→Conv→Dense)"]
    accs = [ann_accuracy, cnn_acc * 100]
    ann_params = build_model(units=(256, 128, 64)).count_params()
    cnn_params = cnn.count_params()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("ANN vs CNN Benchmark", fontsize=14, fontweight="bold")

    colors = ["#3b82f6", "#ef4444"]
    bars = axes[0].bar(labels, accs, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_ylim(96, 100)
    axes[0].set_title("Accuracy")
    for bar, acc in zip(bars, accs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2f}%",
            ha="center",
            fontweight="bold",
        )
    axes[0].grid(axis="y", alpha=0.3)

    p_bars = axes[1].bar(
        labels, [ann_params, cnn_params], color=colors, edgecolor="white", linewidth=1.5
    )
    axes[1].set_ylabel("Parameter Count")
    axes[1].set_title("Model Complexity")
    for bar, p in zip(p_bars, [ann_params, cnn_params]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{p:,}",
            ha="center",
            fontsize=9,
        )
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Benchmark comparison → {save_path}")
    return cnn_acc * 100


# ──────────────────────────────────────────────────────────────
# 8. Hyperparameter Analysis
# ──────────────────────────────────────────────────────────────
def hyperparameter_analysis(
    x_train, y_train_ohe, x_test, y_test_ohe, save_path="figures/hyperparameter_analysis.png"
):
    """
    Sweeps over batch sizes and dropout rates to show their effect
    on test accuracy.
    """
    print("\n" + "=" * 60)
    print("  Hyperparameter Sensitivity Analysis")
    print("=" * 60)

    # Batch size sweep
    batch_sizes = [16, 32, 64, 128, 256]
    batch_accs = []
    for bs in batch_sizes:
        m = build_model()
        m = compile_model(m)
        train_model(m, x_train, y_train_ohe, epochs=8, batch_size=bs, verbose=0)
        _, acc = m.evaluate(x_test, y_test_ohe, verbose=0)
        batch_accs.append(acc * 100)
        print(f"   Batch {bs:>3}: {acc * 100:.2f}%")

    # Dropout sweep
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    dropout_accs = []
    for dr in dropout_rates:
        m = build_model(dropout_rate=dr)
        m = compile_model(m)
        train_model(m, x_train, y_train_ohe, epochs=8, verbose=0)
        _, acc = m.evaluate(x_test, y_test_ohe, verbose=0)
        dropout_accs.append(acc * 100)
        print(f"   Dropout {dr}: {acc * 100:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=14, fontweight="bold")

    ax1.plot(
        [str(b) for b in batch_sizes], batch_accs, "o-", color="#3b82f6", linewidth=2, markersize=8
    )
    ax1.set_title("Batch Size vs Accuracy")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    for x, y in zip(range(len(batch_sizes)), batch_accs):
        ax1.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )

    ax2.plot(
        [str(d) for d in dropout_rates],
        dropout_accs,
        "s-",
        color="#10b981",
        linewidth=2,
        markersize=8,
    )
    ax2.set_title("Dropout Rate vs Accuracy")
    ax2.set_xlabel("Dropout Rate")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    for x, y in zip(range(len(dropout_rates)), dropout_accs):
        ax2.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Hyperparameter analysis → {save_path}")


# ──────────────────────────────────────────────────────────────
# 9. Main Pipeline
# ──────────────────────────────────────────────────────────────
def main():
    # We disable autolog() and defer MLflow completely to the end
    # of the script to prevent the Keras 3 macOS SQLite deadlock.

    print("=" * 60)
    print("  MNIST ANN Classifier — Full Evaluation Pipeline (MLflow Enabled)")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train_flat, x_test_flat, y_train_ohe, y_test_ohe = preprocess_data(
        x_train, x_test, y_train, y_test
    )
    print(f"\n  Train: {x_train_flat.shape}  Test: {x_test_flat.shape}")

    # ── Train or Load ─────────────────────────────────────────
    model = load_saved_model()
    train_loss, train_accuracy = None, None
    if model is None:
        print("\n🚀 No saved model found — training from scratch …\n")
        model = build_model(units=(128, 64), dropout_rate=0.2)
        model = compile_model(model)
        model.summary()
        history = train_model(
            model, x_train_flat, y_train_ohe, epochs=EPOCHS, batch_size=BATCH_SIZE
        )
        save_model(model)
        plot_training_history(history)

        # Save metrics for later MLflow logging
        train_loss = history.history["loss"][-1]
        train_accuracy = history.history["accuracy"][-1]
    else:
        print("⚡ Skipping training — using saved model.")

    # ── Core Evaluation ───────────────────────────────────────
    test_loss, test_acc = model.evaluate(x_test_flat, y_test_ohe, verbose=0)

    print(f"\n{'=' * 60}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"{'=' * 60}")

    # ── Confusion Matrix ──────────────────────────────────────
    plot_confusion_matrix(model, x_test_flat, y_test)

    # ── Per-Class Report ──────────────────────────────────────
    print_classification_report(model, x_test_flat, y_test)

    # ── Sample Predictions ────────────────────────────────────
    plot_sample_predictions(model, x_test_flat, x_test, y_test)

    # ── Export Weights for Web App ────────────────────────────
    export_weights_to_json(model)

    # ── Architecture Comparison ───────────────────────────────
    arch_results = compare_architectures(x_train_flat, y_train_ohe, x_test_flat, y_test_ohe)
    best_acc = max(r["accuracy"] for r in arch_results.values())

    # ── CNN Benchmark ─────────────────────────────────────────
    benchmark_cnn(x_train, y_train_ohe, x_test, y_test_ohe, best_acc)

    # ── Hyperparameter Analysis ───────────────────────────────
    hyperparameter_analysis(x_train_flat, y_train_ohe, x_test_flat, y_test_ohe)

    print("\n" + "=" * 60)
    print("  ✅ All steps completed. Check generated PNG files. Logging to MLflow...")
    print("=" * 60 + "\n")

    # ── Deferred MLflow Logging ───────────────────────────────
    # We log everything at the very end to prevent macOS SQLite thread deadlocks
    import mlflow

    mlflow.set_experiment("MNIST-ANN-Classifier")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "units": (128, 64),
                "dropout_rate": 0.2,
                "random_seed": RANDOM_SEED,
            }
        )
        if train_loss is not None:
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("train_accuracy", train_accuracy)

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        if os.path.exists("figures"):
            mlflow.log_artifacts("figures", artifact_path="evaluation_figures")


if __name__ == "__main__":
    main()
