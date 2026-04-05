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
    if entry["checkpoint"] and (CFG["paths"]["checkpoint_dir"] / entry["checkpoint"]).exists():
        LOADED[name] = load_trained_model(name)
        print(f"  Loaded {name}")

MODEL_CHOICES = list(LOADED.keys())
DEFAULT_MODEL = CFG["app"]["default_model"] if CFG["app"]["default_model"] in LOADED else MODEL_CHOICES[0]


@torch.no_grad()
def classify(image: Image.Image, model_name: str):
    if image is None:
        return "Upload an image first."
    model = LOADED[model_name]
    tensor = val_transform(image.convert("RGB")).unsqueeze(0).to(CFG["device"])
    prob   = torch.sigmoid(model(tensor)).item()
    label  = "Cancer" if prob > 0.5 else "Normal"
    conf   = prob if prob > 0.5 else 1 - prob
    return (
        f"Prediction: {label}\n"
        f"Confidence: {conf:.1%}\n"
        f"Cancer probability: {prob:.4f}\n"
        f"Model: {model_name}"
    )


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="pil", label="Upload histopathology patch (96x96 .tif)"),
        gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Model"),
    ],
    outputs=gr.Textbox(label="Result", lines=4),
    title="Histopathologic Cancer Detection",
    description="Upload a 96x96 histopathology patch to classify it as Normal or Cancer.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_port=CFG["app"]["server_port"], share=False)
