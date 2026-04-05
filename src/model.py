import torch
import torch.nn as nn
import torchvision.models as models

from src import CFG


# ── Model 1: Custom CNN (baseline) ────────────────────
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Model 2: ResNet-34 ────────────────────────────────
def build_resnet34():
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return model


# ── Model 3: EfficientNet-B0 ──────────────────────────
def build_efficientnet_b0():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = True
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return model


# ── Model 4: DenseNet-121 ─────────────────────────────
def build_densenet121():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 256),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return model


# ── Registry ──────────────────────────────────────────
BUILDERS = {
    "Custom CNN":      CustomCNN,
    "ResNet-34":       build_resnet34,
    "EfficientNet-B0": build_efficientnet_b0,
    "DenseNet-121":    build_densenet121,
}


def build_model(name):
    return BUILDERS[name]()


def load_trained_model(name):
    model = build_model(name)
    ckpt  = CFG["checkpoint_files"].get(name)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint for {name} at {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=CFG["device"], weights_only=True))
    model.to(CFG["device"])
    model.eval()
    return model


def load_all_trained_models():
    loaded = {}
    for name, path in CFG["checkpoint_files"].items():
        if path.exists():
            loaded[name] = load_trained_model(name)
            print(f"  Loaded {name} from {path.name}")
    return loaded
