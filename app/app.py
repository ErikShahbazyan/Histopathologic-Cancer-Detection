import sys
from pathlib import Path

# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import torch
from PIL import Image

from src import CFG
from src.dataset import val_transform
from src.model import load_trained_model

# ── Load models on startup ────────────────────────────
LOADED = {}
for entry in CFG["models"]:
    name = entry["name"]
    ckpt_path = CFG["paths"]["checkpoint_dir"] / entry["checkpoint"]
    if entry["checkpoint"] and ckpt_path.exists():
        LOADED[name] = load_trained_model(name)
        print(f"  Loaded {name}")
    else:
        print(f"  Skipping {name} — checkpoint not found")

MODEL_CHOICES = list(LOADED.keys())
ENSEMBLE_LABEL = "Ensemble (all models)"
ALL_CHOICES = MODEL_CHOICES + [ENSEMBLE_LABEL]

DEFAULT_MODEL = (
    CFG["app"]["default_model"]
    if CFG["app"]["default_model"] in LOADED
    else MODEL_CHOICES[0]
)


# ── Prediction helpers ────────────────────────────────
@torch.no_grad()
def _predict_one(model, tensor) -> float:
    """Return cancer probability for a single pre-processed tensor."""
    return torch.sigmoid(model(tensor)).item()


def _format_bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


# ── Main classify function ────────────────────────────
@torch.no_grad()
def classify(image: Image.Image, model_choice: str):
    if image is None:
        return None, "⚠️  Please upload an image first."

    img_rgb = image.convert("RGB")
    tensor = val_transform(img_rgb).unsqueeze(0).to(CFG["device"])

    lines = []

    if model_choice == ENSEMBLE_LABEL:
        # ── Ensemble: run every loaded model, average probabilities ──
        if not LOADED:
            return img_rgb, "No models are loaded."

        probs = {}
        for name, model in LOADED.items():
            probs[name] = _predict_one(model, tensor)

        avg_prob = sum(probs.values()) / len(probs)
        label    = "🔴 Cancer" if avg_prob > 0.5 else "🟢 Normal"
        conf     = avg_prob if avg_prob > 0.5 else 1 - avg_prob

        lines.append("=" * 38)
        lines.append("  ENSEMBLE PREDICTION")
        lines.append("=" * 38)
        lines.append(f"  Result     : {label}")
        lines.append(f"  Confidence : {conf:.1%}")
        lines.append(f"  Avg cancer prob: {avg_prob:.4f}")
        lines.append(f"  {_format_bar(avg_prob)}  {avg_prob:.2%}")
        lines.append("")
        lines.append("── Individual model probabilities ──")
        for name, p in probs.items():
            vote = "Cancer" if p > 0.5 else "Normal"
            lines.append(f"  {name:<18} {p:.4f}  [{vote}]")

    else:
        # ── Single model ──────────────────────────────────────────────
        model = LOADED[model_choice]
        prob  = _predict_one(model, tensor)
        label = "🔴 Cancer" if prob > 0.5 else "🟢 Normal"
        conf  = prob if prob > 0.5 else 1 - prob

        lines.append("=" * 38)
        lines.append(f"  MODEL: {model_choice}")
        lines.append("=" * 38)
        lines.append(f"  Result          : {label}")
        lines.append(f"  Confidence      : {conf:.1%}")
        lines.append(f"  Cancer prob     : {prob:.4f}")
        lines.append(f"  {_format_bar(prob)}  {prob:.2%}")

    return img_rgb, "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────
# with gr.Blocks(title="Histopathologic Cancer Detection") as demo:
with gr.Blocks(
    title="Histopathologic Cancer Detection",
    css="""
    .dark-violet-button {
        background-color: #4B0082 !important;
        color: white !important;
        border: none !important;
    }

    .dark-violet-button:hover {
        background-color: #3a0066 !important;
    }
    """
) as demo:
    gr.Markdown(
        """
        # 🔬 Histopathologic Cancer Detection
        Upload a **96 × 96 histopathology patch** (`.tif` or any image format).
        Choose an individual model **or** select *Ensemble* to combine all three models.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="Input Patch",
                image_mode="RGB",
            )
            model_dd = gr.Dropdown(
                choices=ALL_CHOICES,
                value=ENSEMBLE_LABEL,
                label="Model / Mode",
            )
            # run_btn = gr.Button("🔍 Classify", variant="primary")
            run_btn = gr.Button(
                "🔍 Classify",
                variant="primary",
                elem_classes="dark-violet-button"
            )

        with gr.Column(scale=1):
            img_preview = gr.Image(
                label="Uploaded Image Preview",
                interactive=False,
            )
            result_box = gr.Textbox(
                label="Prediction Result",
                lines=12,
                interactive=False,
            )

    run_btn.click(
        fn=classify,
        inputs=[img_input, model_dd],
        outputs=[img_preview, result_box],
    )

    # Also run on image upload automatically
    img_input.change(
        fn=classify,
        inputs=[img_input, model_dd],
        outputs=[img_preview, result_box],
    )

    gr.Markdown(
        "_Models: ResNet-34 · EfficientNet-B0 · DenseNet-121  |  "
        "Ensemble averages sigmoid probabilities across all loaded models._"
    )


if __name__ == "__main__":
    demo.launch(server_port=CFG["app"]["server_port"], share=False)
