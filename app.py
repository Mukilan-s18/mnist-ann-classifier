"""
MNIST ANN Classifier — Interactive Gradio Demo
================================================
Draw any digit (0–9) on the canvas, click Submit,
and instantly see:
  • Predicted digit with confidence score
  • Full probability bar chart for all 10 classes
  • Saliency heatmap showing which pixels influenced the prediction
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import gradio as gr
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow import keras

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
    784-element float vector ready for the model, perfectly
    matching the official MNIST dataset centering algorithm.
    """
    # 1. Extract stroke data
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        # If RGBA, the alpha channel perfectly isolates the strokes (Alpha=255) from the transparent background (Alpha=0)
        arr = image_array[:, :, 3].astype("float32")
    else:
        # Fallback for RGB/L
        img = Image.fromarray(image_array.astype(np.uint8)).convert("L")
        arr = np.array(img, dtype="float32")
        # Auto-invert if the background is white
        if arr[0, 0] > 127:
            arr = 255.0 - arr

    # 3. Normalize intensity to [0, 1] (fixes dim strokes)
    if arr.max() > 0:
        arr = arr / arr.max()

    # 4. Find bounding box of the drawing
    mask = arr > 0.1
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 784), dtype="float32")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = arr[rmin:rmax+1, cmin:cmax+1]

    # 5. Resize so the maximum dimension is 20 pixels (MNIST standard)
    h, w = cropped.shape
    max_dim = max(h, w)
    scale = 20.0 / max_dim
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    cropped_img = cropped_img.resize((new_w, new_h), Image.LANCZOS)
    cropped_arr = np.array(cropped_img, dtype="float32") / 255.0

    # 6. Center of mass calculation
    y_coords, x_coords = np.indices(cropped_arr.shape)
    total_mass = cropped_arr.sum()
    if total_mass > 0:
        cy = (y_coords * cropped_arr).sum() / total_mass
        cx = (x_coords * cropped_arr).sum() / total_mass
    else:
        cy, cx = new_h / 2.0, new_w / 2.0

    # 7. Place into a 28x28 canvas such that center of mass is at (14, 14)
    canvas = np.zeros((28, 28), dtype="float32")
    start_y = int(np.round(14.0 - cy))
    start_x = int(np.round(14.0 - cx))

    # Clamp bounds just in case
    start_y = max(0, min(28 - new_h, start_y))
    start_x = max(0, min(28 - new_w, start_x))

    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = cropped_arr

    return canvas.reshape(1, 784)


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
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
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
        layers = image.get("layers", [])
        if layers and len(layers) > 0 and layers[0] is not None:
            # The layer contains ONLY the user's transparent drawing, ignoring the background grid!
            arr = layers[0]
        else:
            arr = image.get("composite", image.get("background", None))
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
.sketch-bg {
    background-size: 30px 30px !important;
    background-image:
        linear-gradient(to right, #1e293b 2px, transparent 2px),
        linear-gradient(to bottom, #1e293b 2px, transparent 2px) !important;
    background-color: #000000 !important; /* Change this hex code for different colors! */
    border: 2px solid #334155 !important;
    border-radius: 8px !important;
}
/* Force the internal Gradio canvas to be transparent so our dark background shows through */
.sketch-bg canvas, .sketch-bg .image-container, .sketch-bg .wrap, .sketch-bg .canvas-wrap, .sketch-bg .image-frame {
    background-color: transparent !important;
}
/* Hide the built-in Gradio trash can icon to force use of the global Clear button */
.sketch-bg button[aria-label="Clear"],
.sketch-bg button[aria-label="Remove Image"],
.sketch-bg button[aria-label="Remove"],
.sketch-bg button[title="Clear"],
.sketch-bg button[title="Remove"] {
    display: none !important;
}
"""

JS_DARK_MODE = """
function() {
    document.body.classList.add('dark');
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

with gr.Blocks(title="MNIST ANN Classifier", css=CSS, js=JS_DARK_MODE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                label="✏️ Draw Here",
                type="numpy",
                image_mode="RGBA",
                height=300,
                width=300,
                elem_classes=["sketch-bg"],
                brush=gr.Brush(default_size=20, colors=["#ff0000", "#ffffff", "#000000"], default_color="#ff0000"),
            )
            gr.Markdown(EXAMPLES_NOTE)
            with gr.Row():
                submit_btn = gr.Button("🔍 Predict", variant="primary")
                clear_btn  = gr.Button("🗑️ Clear")

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

    clear_btn.click(
        fn=lambda: (None, None, None, None),
        inputs=None,
        outputs=[canvas, prediction_label, prob_chart, saliency_plot],
    )

    gr.Markdown("""
---
**Model Details** | Trained on MNIST (60k images) | Optimizer: Adam | Epochs: 15 with Early Stopping
| Test Accuracy: **97.86%** | [View Source Code](https://github.com/Mukilan-s18/mnist-ann-classifier)
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, css=CSS)
