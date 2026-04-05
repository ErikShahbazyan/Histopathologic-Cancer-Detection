import numpy as np
import pandas as pd
from PIL import Image

import torch

from src import CFG
from src.dataset import get_test_loader, val_transform
from src.model import load_trained_model


@torch.no_grad()
def predict_test(model, loader):
    model.eval()
    model.to(CFG["device"])
    all_ids, all_probs = [], []
    for imgs, ids in loader:
        preds = torch.sigmoid(model(imgs.to(CFG["device"])))
        all_probs.extend(preds.cpu().numpy().flatten())
        all_ids.extend(ids)
    return all_ids, all_probs


def generate_submission(model, output_name="submission.csv"):
    test_loader = get_test_loader()
    ids, probs  = predict_test(model, test_loader)
    submission  = pd.DataFrame({"id": ids, "label": probs})
    output_path = CFG["paths"]["output_dir"] / output_name
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission.head())
    return submission


@torch.no_grad()
def predict_single(image_path, model_name=None):
    if model_name is None:
        model_name = CFG["app"]["default_model"]
    model = load_trained_model(model_name)

    img = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(CFG["device"])
    logit = model(tensor)
    prob  = torch.sigmoid(logit).item()

    return {
        "model":       model_name,
        "probability": prob,
        "prediction":  "Cancer" if prob > 0.5 else "Normal",
        "confidence":  prob if prob > 0.5 else 1 - prob,
    }
