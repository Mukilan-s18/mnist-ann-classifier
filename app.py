"""
MNIST ANN Classifier — Interactive Gradio Demo
================================================
Draw any digit (0–9) on the canvas, click Submit,
and instantly see:
  • Predicted digit with confidence score
  • Full probability bar chart for all 10 classes
  • Saliency heatmap showing which pixels influenced the prediction
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import gradio as gr
from tensorflow import keras
import tensorflow as tf
import os

# ── Load model ────────────────────────────────────────────────
MODEL_PATH = "saved_model/mnist_ann.keras"
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

DIGIT_LABELS = [str(i) for i in range(10)]
EMOJI = ["0️⃣","1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣"]


# ── Preprocessing ─────────────────────────────────────────────
def preprocess(image_array: np.ndarray) -> np.ndarray:
    """
    Converts a raw RGBA/RGB canvas image into a normalised
    784-element float vector ready for the model.
    """
    # Convert to PIL, resize to 28×28 greyscale
    img = Image.fromarray(image_array.astype(np.uint8))
    img = img.convert("L")          # greyscale
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype="float32")

    # Gradio sketchpad returns white-on-black; MNIST is also white-on-black
    arr = arr / 255.0
    return arr.reshape(1, 784)


# ── Saliency map ──────────────────────────────────────────────
def compute_saliency(input_vector: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Computes vanilla gradient saliency: how much does each input pixel
    affect the predicted class score?

    Returns a 28×28 normalised heatmap.
    """
    # Keras 3 requires explicit tape.watch on a tf.constant
    x = tf.constant(input_vector, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        score = preds[:, class_idx]

    grads = tape.gradient(score, x)          # shape (1, 784)

    if grads is None:
        # Fallback: return uniform heatmap if gradients unavailable
        return np.ones((28, 28), dtype="float32") * 0.5

    saliency = tf.abs(grads).numpy().reshape(28, 28)

    # Normalise to [0, 1]
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    return saliency


def make_saliency_figure(original_28: np.ndarray,
                         saliency: np.ndarray,
                         pred_digit: int,
                         confidence: float) -> plt.Figure:
    """
    Creates a side-by-side figure: original digit | saliency overlay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Left: original
    axes[0].imshow(original_28, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input", color="white", fontsize=11)
    axes[0].axis("off")

    # Right: saliency overlay
    axes[1].imshow(original_28, cmap="gray", vmin=0, vmax=1, alpha=0.4)
    axes[1].imshow(saliency, cmap="hot", alpha=0.7, vmin=0, vmax=1)
    axes[1].set_title(
        f"Saliency — why {EMOJI[pred_digit]} ?",
        color="white", fontsize=11
    )
    axes[1].axis("off")

    fig.suptitle(
        f"Predicted: {EMOJI[pred_digit]}  ({confidence*100:.1f}% confidence)",
        color="#38bdf8", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(pad=0.5)
    return fig


# ── Main inference function ───────────────────────────────────
def predict(image):
    """
    Called by Gradio on every Submit. Returns:
      - label string
      - {class: probability} dict for bar chart
      - saliency matplotlib figure
    """
    if image is None:
        return "Draw a digit first!", {str(i): 0.0 for i in range(10)}, None

    # Handle both dict (sketchpad) and ndarray inputs
    if isinstance(image, dict):
        arr = image.get("composite", image.get("layers", [None])[0])
    else:
        arr = image

    if arr is None:
        return "Draw a digit first!", {str(i): 0.0 for i in range(10)}, None

    # Preprocess
    x = preprocess(arr)
    original_28 = x.reshape(28, 28)

    # Inference
    probs     = model.predict(x, verbose=0)[0]          # shape (10,)
    pred_idx  = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # Saliency
    saliency = compute_saliency(x, pred_idx)
    fig      = make_saliency_figure(original_28, saliency, pred_idx, confidence)

    # Label string
    label = f"{EMOJI[pred_idx]}  Digit {pred_idx}  — {confidence*100:.1f}% confident"

    # Probabilities dict for Gradio bar chart
    prob_dict = {
        f"{EMOJI[i]} Digit {i}": float(probs[i]) for i in range(10)
    }

    return label, prob_dict, fig


# ── Gradio UI ─────────────────────────────────────────────────
CSS = """
body { background: #0f172a !important; }
.gradio-container {
    background: #0f172a !important;
    font-family: 'Inter', sans-serif;
}
h1 { color: #38bdf8 !important; text-align: center; }
.gr-button-primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}
.gr-button-secondary {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
}
.output-label {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #38bdf8 !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    text-align: center !important;
}
"""

TITLE = "✍️ MNIST Digit Classifier — ANN with Saliency Maps"

DESCRIPTION = """
## Draw any digit (0–9) on the canvas below and hit **Submit**

This fully connected neural network (ANN) was trained on the MNIST dataset and achieves **97.86% test accuracy**.

🔥 The **Saliency Map** shows *exactly which pixels* drove the prediction — no black box!

> Architecture: `Input(784) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(10, Softmax)`
"""

EXAMPLES_NOTE = """
**Tips for best results:**
- Draw the digit **large and centered**
- Use thick strokes
- White digit on dark background works best
"""

with gr.Blocks(css=CSS, title="MNIST ANN Classifier") as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                label="✏️ Draw Here",
                type="numpy",
                image_mode="RGB",
                height=300,
                width=300,
                brush=gr.Brush(default_size=20, colors=["#ffffff"], default_color="#ffffff"),
            )
            gr.Markdown(EXAMPLES_NOTE)
            with gr.Row():
                submit_btn = gr.Button("🔍 Predict", variant="primary")
                clear_btn  = gr.ClearButton([canvas], value="🗑️ Clear")

        with gr.Column(scale=1):
            prediction_label = gr.Label(
                label="🎯 Prediction",
                num_top_classes=3,
            )
            prob_chart = gr.Label(
                label="📊 All Class Probabilities",
                num_top_classes=10,
            )

    saliency_plot = gr.Plot(label="🔥 Saliency Map — What the model is looking at")

    submit_btn.click(
        fn=predict,
        inputs=[canvas],
        outputs=[prediction_label, prob_chart, saliency_plot],
    )

    gr.Markdown("""
---
**Model Details** | Trained on MNIST (60k images) | Optimizer: Adam | Epochs: 15 with Early Stopping
| Test Accuracy: **97.86%** | [View Source Code](https://github.com/Mukilan-s18/mnist-ann-classifier)
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
